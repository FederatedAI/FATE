#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging

from typing import Union

from arch.api.base.federation import Federation, Party, Rubbish
from arch.api.utils import file_utils
from arch.api.utils import log_utils
from arch.api.impl.based_spark.based_hdfs.rabbit_manager import RabbitManager
from arch.api.impl.based_spark.based_hdfs.table import RDDTable
from arch.api.impl.based_spark.based_hdfs.mq_channel import MQChannel
from arch.api.impl.based_spark import util
from functools import partial
import pika
from pickle import dumps as p_dumps, loads as p_loads
import json
from pyspark.rdd import RDD
from pyspark import SparkContext

LOGGER = log_utils.getLogger()


class FederationRuntime(Federation):
    def __init__(self, session_id, runtime_conf):
        super().__init__(session_id, runtime_conf)
        LOGGER.debug("runtime_conf:{}.".format(runtime_conf))
        self._role = runtime_conf.get("local").get("role")
        self._party_id = str(runtime_conf.get("local").get("party_id"))
        self._session_id = session_id
        #init mq here
        self.init_mq_context(runtime_conf)

    def init_mq_context(self, runtime_conf):
        _path = file_utils.get_project_base_directory() + "/arch/conf/server_conf.json"
        server_conf = file_utils.load_json_conf(_path)
        self._mq_conf = server_conf.get('servers').get('rabbitmq')
        self._self_mq = {}
        self._self_mq["host"] = self._mq_conf.get(self._party_id).get("host")
        self._self_mq["port"] = self._mq_conf.get(self._party_id).get("port")
        self._mng_port = self._mq_conf.get(self._party_id).get("mng_port")
        base_user = self._mq_conf.get(self._party_id).get('user')
        base_password = self._mq_conf.get(self._party_id).get('password')

        self._rabbit_manager = RabbitManager(base_user, base_password, "{}:{}".format(self._self_mq["host"], self._mng_port))

        federation_info = runtime_conf.get("job_parameters", {}).get("federation_info", {})
        self._self_mq['union_name'] = federation_info.get('union_name')
        self._self_mq['policy_id'] = federation_info.get("policy_id")

        # initial user
        self._rabbit_manager.CreateUser(self._self_mq['union_name'], self._self_mq['policy_id'])

        self._queque_map = {}
        self._channels_map = {}

    def gen_vhost_name(self, party_id):
        parties = [self._party_id, party_id]
        parties.sort()
        union_name = "{}-{}".format(parties[0], parties[1])
        vhost_name = "{}-{}".format(self._session_id, union_name)
        return union_name, vhost_name

    def gen_names(self, party_id):
        names = {}
        union_name, vhost = self.gen_vhost_name(party_id)
        names["vhost"] = vhost
        names["union"] = union_name
        names["send"] = "{}-{}".format("send", vhost)
        names["receive"] = "{}-{}".format("receive", vhost)
        return names

    def get_mq_names(self, parties: Union[Party, list]):
        if isinstance(parties, Party):
            parties = [parties]
        party_ids = [str(party.party_id) for party in parties]
        mq_names = {}
        for party_id in party_ids:
            LOGGER.debug("get_mq_names, party_id={}, self._mq_conf={}.".format(party_id, self._mq_conf))
            names = self._queque_map.get(party_id)
            if names is None:
                names = self.gen_names(party_id)

                # initial vhost
                self._rabbit_manager.CreateVhost(names["vhost"])
                self._rabbit_manager.AddUserToVhost(self._self_mq['union_name'], names["vhost"])

                # initial send queue, the name is send-${vhost}
                self._rabbit_manager.CreateQueue(names["vhost"], names["send"])

                # initial receive queue, the name is receive-${vhost}
                self._rabbit_manager.CreateQueue(names["vhost"], names["receive"])

                host = self._mq_conf.get(party_id).get("host")
                port = self._mq_conf.get(party_id).get("port")
                
                upstream_uri = "amqp://{}:{}@{}:{}".format(self._self_mq['union_name'], self._self_mq['policy_id'], host, port)
                self._rabbit_manager.FederateQueue(upstream_host=upstream_uri, vhost=names["vhost"], union_name=names["union"])

                self._queque_map[party_id] = names
            mq_names[party_id] = names
        LOGGER.debug("get_mq_names:{}".format(mq_names))
        return mq_names

    def generate_mq_names(self, parties: Union[Party, list]):
        if isinstance(parties, Party):
            parties = [parties]
        party_ids = [str(party.party_id) for party in parties]
        for party_id in party_ids:
            LOGGER.debug("generate_mq_names, party_id={}, self._mq_conf={}.".format(party_id, self._mq_conf))
            names = self.gen_names(party_id)
            self._queque_map[party_id] = names
        LOGGER.debug("generate_mq_names:{}".format(self._queque_map))

    def get_channels(self, mq_names, host, port, user, password):
        LOGGER.debug("mq_names:{}.".format(mq_names))
        channel_infos = []
        for party_id, names in mq_names.items():
            info = self._channels_map.get(party_id)
            if info is None:
                info = FederationRuntime._get_channel(host, port, user, password, names, party_id)
                self._channels_map[party_id] = info
            channel_infos.append(info)
        LOGGER.debug("got channel_infos.")
        return channel_infos

    @staticmethod
    def _get_channel(host, port, user, password, names, party_id):
        return MQChannel(host=host, port=port, user=user, password=password, party_id=party_id,
                         vhost=names["vhost"], send_queue_name=names["send"], receive_queue_name=names["receive"])


    @staticmethod
    def _send_kv(name, tag, data, channel_infos, total_size, partitions):
        headers={"total_size":total_size, "partitions":partitions}
        for info in channel_infos:
            properties=pika.BasicProperties(
                content_type='application/json',
                app_id=info._party_id,
                message_id=name,
                correlation_id=tag,
                headers=headers
            )
            LOGGER.debug("_send_kv, info:{}, properties:{}.".format(info, properties))
            info.basic_publish(body=json.dumps(data), properties=properties)

    @staticmethod
    def _send_obj(name, tag, data, channel_infos):
        for info in channel_infos:
            properties=pika.BasicProperties(
                content_type='text/plain',
                app_id=info._party_id,
                message_id=name,
                correlation_id=tag
            )
            LOGGER.debug("_send_obj, properties:{}.".format(properties))
            info.basic_publish(body=data, properties=properties)

    # can't pickle _thread.lock objects
    @staticmethod
    def _get_channels(mq_names, host, port, user, password):
        channel_infos = []
        for party_id, names in mq_names.items():
            info = FederationRuntime._get_channel(host, port, user, password, names, party_id)
            channel_infos.append(info)
        return channel_infos

    @staticmethod
    def _partition_send(kvs, name, tag, total_size, partitions, mq_names, self_mq):
        LOGGER.debug("_partition_send, total_size:{}, partitions:{}, mq_names:{}, self_mq:{}.".format(total_size, partitions, mq_names, self_mq))
        channel_infos = FederationRuntime._get_channels(mq_names=mq_names, host=self_mq["host"], port=self_mq["port"], 
                                                        user=self_mq['union_name'], password=self_mq['policy_id'])
        data = []
        lines = 0
        MESSAGE_MAX_SIZE = 200000
        for k, v in kvs:
            el = {}
            el['k'] = p_dumps(k).hex()
            el['v'] = p_dumps(v).hex()
            data.append(el)
            lines = lines + 1
            if lines > MESSAGE_MAX_SIZE:
                FederationRuntime._send_kv(name=name, tag=tag, data=data, channel_infos=channel_infos, total_size=total_size, partitions=partitions)
                lines = 0
                data.clear()
        FederationRuntime._send_kv(name=name, tag=tag, data=data, channel_infos=channel_infos, total_size=total_size, partitions=partitions)
        return data

    @staticmethod
    def _receive(channel_info, name, tag):
        count = 0
        obj = None
        for method, properties, body in channel_info.consume():
            LOGGER.debug("_receive, count:{}, method:{}, properties:{}.".format(count, method, properties))
            if properties.message_id == name and properties.correlation_id == tag:
                if properties.content_type == 'text/plain':
                    obj = p_loads(body)
                    channel_info.basic_ack(delivery_tag=method.delivery_tag)
                    break
                elif properties.content_type == 'application/json':
                    data = json.loads(body)
                    count += len(data)
                    data_iter = ( (p_loads(bytes.fromhex(el['k'])), p_loads(bytes.fromhex(el['v']))) for el in data)
                    sc = SparkContext.getOrCreate()
                    if obj:
                        rdd = sc.parallelize(data_iter, properties.headers["partitions"])
                        obj = obj.union(rdd)
                        LOGGER.debug("before coalesce: federation get union partition %d, count: %d" % (obj.getNumPartitions(), obj.count()))
                        obj = obj.coalesce(properties.headers["partitions"])
                        LOGGER.debug("end coalesce: federation get union partition %d, count: %d" % (obj.getNumPartitions(), obj.count()))
                    else:
                        obj = sc.parallelize(data_iter, properties.headers["partitions"]).persist(util.get_storage_level())
                    if count == properties.headers["total_size"]:
                        channel_info.basic_ack(delivery_tag=method.delivery_tag)
                        break

                channel_info.basic_ack(delivery_tag=method.delivery_tag)
        # return any pending messages
        channel_info.cancel()
        return obj

    def get(self, name, tag, parties: Union[Party, list]):
        LOGGER.debug("start to get obj, name={}, tag={}, parties={}.".format(name, tag, parties))
        rubbish = Rubbish(name=name, tag=tag)
        mq_names = self.get_mq_names(parties)
        channel_infos = self.get_channels(mq_names=mq_names, host=self._self_mq["host"], port=self._self_mq["port"], 
                                            user=self._self_mq['union_name'], password=self._self_mq['policy_id'])
        
        rtn = []
        for info in channel_infos:
            obj = FederationRuntime._receive(info, name, tag)
            LOGGER.info(f'federation got data. name: {name}, tag: {tag}')
            if isinstance(obj, RDD):
                rtn.append(obj)
                # rubbish.add_table(obj)
            else:
                rtn.append(obj)
        LOGGER.debug("finish get obj, name={}, tag={}, parties={}.".format(name, tag, parties))
        return rtn, rubbish      

    def remote(self, obj, name, tag, parties: Union[Party, list]):
        LOGGER.debug("start to remote obj, name={}, tag={}, parties={}.".format(name, tag, parties))
        rubbish = Rubbish(name=name, tag=tag)
        mq_names = self.get_mq_names(parties)

        if isinstance(obj, RDD):
            total_size=obj.count()
            partitions=obj.getNumPartitions()
            LOGGER.debug("start to remote RDD, total_size={}, partitions={}.".format(total_size, partitions))
            send_func = partial(FederationRuntime._partition_send, name=name, tag=tag, 
                                total_size=total_size, partitions=partitions, mq_names=mq_names, self_mq=self._self_mq)
            obj.mapPartitions(send_func).collect()
            # rubbish.add_table(obj)
        else:
            channel_infos = self.get_channels(mq_names=mq_names, host=self._self_mq["host"], port=self._self_mq["port"], 
                                          user=self._self_mq['union_name'], password=self._self_mq['policy_id'])
            FederationRuntime._send_obj(name=name, tag=tag, data=p_dumps(obj), channel_infos=channel_infos)
        LOGGER.debug("finish remote obj, name={}, tag={}, parties={}.".format(name, tag, parties))
        return rubbish
    
    def cleanup(self):
        LOGGER.debug("federation start to cleanup...")
        for party_id, names in self._queque_map.items():
            LOGGER.debug("cleanup partyid={}, names={}.".format(party_id, names))
            self._rabbit_manager.DeFederateQueue(union_name=names["union"], vhost=names["vhost"])
            self._rabbit_manager.DeleteQueue(vhost=names["vhost"], queue_name=names["send"])
            self._rabbit_manager.DeleteQueue(vhost=names["vhost"], queue_name=names["receive"])
            self._rabbit_manager.DeleteVhost(vhost=names["vhost"])
        self._queque_map.clear()
        if self._self_mq['union_name']:
            LOGGER.debug("clean user {}.".format(self._self_mq['union_name']))
            self._rabbit_manager.DeleteUser(user=self._self_mq['union_name'])

