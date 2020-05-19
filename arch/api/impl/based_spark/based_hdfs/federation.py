import logging

from typing import Union

from arch.api.base.federation import Federation, Party, Rubbish
from arch.api.utils import file_utils
from arch.api.utils import log_utils
from arch.api.impl.based_spark.based_hdfs.rabbit_manager import RabbitManager
from arch.api.impl.based_spark.based_hdfs.table import RDDTable
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
        self._host = self._mq_conf.get(self._party_id).get("host")
        self._port = self._mq_conf.get(self._party_id).get("port")
        self._mng_port = self._mq_conf.get(self._party_id).get("mng_port")
        base_user = self._mq_conf.get(self._party_id).get("user")
        base_password = self._mq_conf.get(self._party_id).get("password")

        self._rabbit_manager = RabbitManager(base_user, base_password, "{}:{}".format(self._host, self._mng_port))

        mq_info = runtime_conf.get("job_parameters", {}).get("mq_info", {})
        self._user = mq_info.get("user")
        self._password = mq_info.get("pswd")

        # initial user
        self._rabbit_manager.CreateUser(self._user, self._password)

        self._queque_map = {}
        self._channels_map = {}


    def gen_vhost_name(self, party_id):
        parties = [self._party_id, party_id]
        parties.sort()
        union_name = "{}-{}".format(parties[0], parties[1])
        vhost_name = "{}-{}".format(self._session_id, union_name)
        return union_name, vhost_name


    def get_mq_names(self, parties: Union[Party, list]):
        if isinstance(parties, Party):
            parties = [parties]
        party_ids = [str(party.party_id) for party in parties]
        mq_names = {}
        for party_id in party_ids:
            LOGGER.debug("get_mq_names, party_id={}, self._mq_conf={}.".format(party_id, self._mq_conf))
            names = self._queque_map.get(party_id)
            if names is None:
                names = {}

                # initial vhost
                union_name, vhost = self.gen_vhost_name(party_id)
                self._rabbit_manager.CreateVhost(vhost)
                self._rabbit_manager.AddUserToVhost(self._user, vhost)
                names["vhost"] = vhost

                # initial send queue, the name is send-${vhost}
                send_queue_name = "{}-{}".format("send", vhost)
                self._rabbit_manager.CreateQueue(vhost, send_queue_name)
                names["send"] = send_queue_name

                # initial receive queue, the name is receive-${vhost}
                receive_queue_name = "{}-{}".format("receive", vhost)
                self._rabbit_manager.CreateQueue(vhost, receive_queue_name)
                names["receive"] = receive_queue_name

                host = self._mq_conf.get(party_id).get("host")
                port = self._mq_conf.get(party_id).get("port")
                
                upstream_uri = "amqp://{}:{}@{}:{}".format(self._user, self._password, host, port)
                union_name = self._rabbit_manager.FederateQueue(upstream_host=upstream_uri, vhost=vhost, union_name=union_name)
                names["union"] = union_name

                self._queque_map[party_id] = names
            mq_names[party_id] = names
        LOGGER.debug("get_mq_names:{}".format(mq_names))
        return mq_names


    def get_channels(self, mq_names, host, port, user, password):
        channel_infos = []
        for party_id, names in mq_names.items():
            info = self._channels_map.get(party_id)
            if info is None:
                info = FederationRuntime.get_channel(host, port, user, password, names, party_id)
                self._channels_map[party_id] = info
            channel_infos.append(info)
        return channel_infos
    

    @staticmethod
    def get_channel(host, port, user, password, names, party_id):
        credentials = pika.PlainCredentials(user, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host, port, names["vhost"], credentials))
        info= {}
        info["channel"] = connection.channel()
        info["send"] = names["send"]
        info["receive"] = names["receive"]
        info["party_id"] = party_id
        return info


    @staticmethod
    def _get_channels(mq_names, host, port, user, password):
        LOGGER.debug("mq_names:{}.".format(mq_names))
        channel_infos = []
        for party_id, names in mq_names.items():
            channel_infos.append(FederationRuntime.get_channel(host, port, user, password, names, party_id))
        LOGGER.debug("channel_infos:{}.".format(channel_infos))
        return channel_infos


    @staticmethod
    def _send_kv(name, tag, data, channel_infos, total_size, partitions):
        headers={"total_size":total_size, "partitions":partitions}
        for info in channel_infos:
            properties=pika.BasicProperties(
                content_type='application/json',
                app_id=info["party_id"],
                message_id=name,
                correlation_id=tag,
                headers=headers
            )
            LOGGER.debug("basic_publish, info:{}, properties:{}, data:{}.".format(info, properties, json.dumps(data)))
            info["channel"].basic_publish(exchange='', routing_key=info["send"], body=json.dumps(data), properties=properties)

    
    @staticmethod
    def _send_obj(name, tag, data, channel_infos):
        for info in channel_infos:
            properties=pika.BasicProperties(
                content_type='text/plain',
                app_id=info["party_id"],
                message_id=name,
                correlation_id=tag
            )
            LOGGER.debug("_send_obj, body:{},  properties:{}.".format(data, properties))
            info["channel"].basic_publish(exchange='', routing_key=info["send"], body=data, properties=properties)



    @staticmethod
    def _partition_send(kvs, mq_names, host, port, user, password, name, tag, total_size, partitions):
        LOGGER.debug("_partition_send, mq_names:{}, total_size:{}, partitions:{}.".format(mq_names, total_size, partitions))
        channel_infos = FederationRuntime._get_channels(mq_names=mq_names, host=host, port=port, user=user, password=password)
        LOGGER.debug("channel_infos:{}.".format(channel_infos))
        data = []
        lines = 0
        MESSAGE_MAX_SIZE = 100 #200000
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
        for method, properties, body in channel_info["channel"].consume(queue=channel_info["receive"]):
            LOGGER.debug("_receive, count:{}, method:{}, properties:{}.".format(count, method, properties))
            if properties.message_id == name and properties.correlation_id == tag:
                if properties.content_type == 'text/plain':
                    obj = p_loads(body)
                    channel_info["channel"].basic_ack(delivery_tag=method.delivery_tag)
                    break
                elif properties.content_type == 'application/json':
                    data = json.loads(body)
                    count += len(data)
                    data_iter = ( (p_loads(bytes.fromhex(el['k'])), p_loads(bytes.fromhex(el['v']))) for el in data)
                    sc = SparkContext.getOrCreate()
                    if obj:
                        rdd = sc.parallelize(data_iter, properties.headers["partitions"])
                        obj = obj.union(rdd)
                    else:
                        obj = sc.parallelize(data_iter, properties.headers["partitions"]).persist(util.get_storage_level())
                    if count == properties.headers["total_size"]:
                        channel_info["channel"].basic_ack(delivery_tag=method.delivery_tag)
                        break

                channel_info["channel"].basic_ack(delivery_tag=method.delivery_tag)
        # return any pending messages
        channel_info["channel"].cancel()
        return obj

    def get(self, name, tag, parties: Union[Party, list]):
        LOGGER.debug("start to get obj, name={}, tag={}, parties={}.".format(name, tag, parties))
        rubbish = Rubbish(name=name, tag=tag)
        mq_names = self.get_mq_names(parties)
        channel_infos = self.get_channels(mq_names=mq_names, host=self._host, port=self._port, 
                                            user=self._user, password=self._password)
        
        rtn = []
        for info in channel_infos:
            obj = FederationRuntime._receive(info, name, tag)
            LOGGER.info(f'federation got data. name: {name}, tag: {tag}')
            if isinstance(obj, RDD):
                rtn.append(obj)
                rubbish.add_table(obj)
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
            send_func = partial(FederationRuntime._partition_send, 
                        mq_names=mq_names, 
                        host=self._host, port=self._port, 
                        user=self._user, password=self._password,
                        name=name, tag=tag, total_size=total_size, partitions=partitions)
            obj.mapPartitions(send_func).collect()
            rubbish.add_table(obj)
        else:
            channel_infos = self.get_channels(mq_names=mq_names, host=self._host, port=self._port, 
                                            user=self._user, password=self._password)
            FederationRuntime._send_obj(name=name, tag=tag, data=p_dumps(obj), channel_infos=channel_infos)
        LOGGER.debug("finish remote obj, name={}, tag={}, parties={}.".format(name, tag, parties))
        return rubbish

            



