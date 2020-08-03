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

import json
import typing
from functools import partial
from pickle import dumps as p_dumps, loads as p_loads

import pika
# noinspection PyPackageRequirements
from pyspark import SparkContext, RDD

from fate_arch.abc import FederationABC, GarbageCollectionABC
from fate_arch.backend.spark import MQChannel
from fate_arch.backend.spark import RabbitManager
from fate_arch.backend.spark import get_storage_level
from fate_arch.common import Party
from fate_arch.common.log import getLogger

LOGGER = getLogger()


class MQ(object):
    def __init__(self, host, port, union_name, policy_id, mq_conf):
        self.host = host
        self.port = port
        self.union_name = union_name
        self.policy_id = policy_id
        self.mq_conf = mq_conf


class FederationEngine(FederationABC):
    def __init__(self, session_id, party: Party, mq: MQ, rabbit_manager: RabbitManager):
        self._session_id = session_id
        self._party = party
        self._mq = mq
        self._rabbit_manager = rabbit_manager

        self._queue_map = {}
        self._channels_map = {}

    def get(self, name, tag, parties: typing.List[Party], gc: GarbageCollectionABC) -> typing.List:
        log_str = f"federation.get(name={name}, tag={tag}, parties={parties})"
        LOGGER.debug(f"[{log_str}]start to get obj")

        mq_names = self._get_mq_names(parties)
        channel_infos = self._get_channels(mq_names=mq_names)

        rtn = []
        for info in channel_infos:
            obj = _receive(info, name, tag)
            LOGGER.info(f'federation got data. name: {name}, tag: {tag}')
            if isinstance(obj, RDD):
                rtn.append(obj)
            else:
                rtn.append(obj)
        LOGGER.debug("finish get obj, name={}, tag={}, parties={}.".format(name, tag, parties))
        return rtn

    def remote(self, obj, name: str, tag: str, parties: typing.List[Party],
               gc: GarbageCollectionABC) -> typing.NoReturn:
        LOGGER.debug("start to remote obj, name={}, tag={}, parties={}.".format(name, tag, parties))
        mq_names = self._get_mq_names(parties)

        if isinstance(obj, RDD):
            total_size = obj.count()
            partitions = obj.getNumPartitions()
            LOGGER.debug("start to remote RDD, total_size={}, partitions={}.".format(total_size, partitions))
            send_func = partial(_partition_send, name=name, tag=tag,
                                total_size=total_size, partitions=partitions, mq_names=mq_names, self_mq=self._mq)
            obj.mapPartitions(send_func).collect()
        else:
            channel_infos = self._get_channels(mq_names=mq_names)
            _send_obj(name=name, tag=tag, data=p_dumps(obj), channel_infos=channel_infos)
        LOGGER.debug("finish remote obj, name={}, tag={}, parties={}.".format(name, tag, parties))

    def cleanup(self):
        LOGGER.debug("federation start to cleanup...")
        for party_id, names in self._queue_map.items():
            LOGGER.debug(f"cleanup party_id={party_id}, names={names}.")
            self._rabbit_manager.de_federate_queue(union_name=names["union"], vhost=names["vhost"])
            self._rabbit_manager.delete_queue(vhost=names["vhost"], queue_name=names["send"])
            self._rabbit_manager.delete_queue(vhost=names["vhost"], queue_name=names["receive"])
            self._rabbit_manager.delete_vhost(vhost=names["vhost"])
        self._queue_map.clear()
        if self._mq.union_name:
            LOGGER.debug(f"clean user {self._mq.union_name}.")
            self._rabbit_manager.delete_user(user=self._mq.union_name)

    def _gen_names(self, party_id):
        names = {}
        left, right = (self._party.party_id, party_id) if self._party.party_id < party_id else (
            party_id, self._party.party_id)
        union_name = f"{left}-{right}"
        vhost_name = f"{self._session_id}-{union_name}"
        names["vhost"] = vhost_name
        names["union"] = union_name
        names["send"] = f"send-{vhost_name}"
        names["receive"] = f"receive-{vhost_name}"
        return names

    def _get_mq_names(self, parties: typing.List[Party]):

        party_ids = [str(party.party_id) for party in parties]
        mq_names = {}
        for party_id in party_ids:
            LOGGER.debug(f"get_mq_names, party_id={party_id}, self._mq={self._mq}.")
            names = self._queue_map.get(party_id)
            if names is None:
                names = self._gen_names(party_id)
                # initial vhost
                self._rabbit_manager.create_vhost(names["vhost"])
                self._rabbit_manager.add_user_to_vhost(self._mq.union_name, names["vhost"])

                # initial send queue, the name is send-${vhost}
                self._rabbit_manager.create_queue(names["vhost"], names["send"])

                # initial receive queue, the name is receive-${vhost}
                self._rabbit_manager.create_queue(names["vhost"], names["receive"])

                host = self._mq.mq_conf.get(party_id).get("host")
                port = self._mq.mq_conf.get(party_id).get("port")

                upstream_uri = f"amqp://{self._mq.union_name}:{self._mq.policy_id}@{host}:{port}"
                self._rabbit_manager.federate_queue(upstream_host=upstream_uri, vhost=names["vhost"],
                                                    union_name=names["union"])

                self._queue_map[party_id] = names
            mq_names[party_id] = names
        LOGGER.debug("get_mq_names:{}".format(mq_names))
        return mq_names

    def _generate_mq_names(self, parties: typing.Union[Party, list]):
        if isinstance(parties, Party):
            parties = [parties]
        party_ids = [str(party.party_id) for party in parties]
        for party_id in party_ids:
            LOGGER.debug("generate_mq_names, party_id={}, self._mq_conf={}.".format(party_id, self._mq.mq_conf))
            names = self._gen_names(party_id)
            self._queue_map[party_id] = names
        LOGGER.debug("generate_mq_names:{}".format(self._queue_map))

    def _get_channels(self, mq_names):
        LOGGER.debug("mq_names:{}.".format(mq_names))
        channel_infos = []
        for party_id, names in mq_names.items():
            info = self._channels_map.get(party_id)
            if info is None:
                info = _get_channel(self._mq, names, party_id)
                self._channels_map[party_id] = info
            channel_infos.append(info)
        LOGGER.debug("got channel_infos.")
        return channel_infos


def _get_channel(mq, names, party_id):
    return MQChannel(host=mq.host, port=mq.port, user=mq.union_name, password=mq.policy_id, party_id=party_id,
                     vhost=names["vhost"], send_queue_name=names["send"], receive_queue_name=names["receive"])


def _send_kv(name, tag, data, channel_infos, total_size, partitions):
    headers = {"total_size": total_size, "partitions": partitions}
    for info in channel_infos:
        properties = pika.BasicProperties(
            content_type='application/json',
            app_id=info.party_id,
            message_id=name,
            correlation_id=tag,
            headers=headers
        )
        LOGGER.debug("_send_kv, info:{}, properties:{}.".format(info, properties))
        info.basic_publish(body=json.dumps(data), properties=properties)


def _send_obj(name, tag, data, channel_infos):
    for info in channel_infos:
        properties = pika.BasicProperties(
            content_type='text/plain',
            app_id=info.party_id,
            message_id=name,
            correlation_id=tag
        )
        LOGGER.debug("_send_obj, properties:{}.".format(properties))
        info.basic_publish(body=data, properties=properties)


# can't pickle _thread.lock objects
def _get_channels(mq_names, mq):
    channel_infos = []
    for party_id, names in mq_names.items():
        info = _get_channel(mq, names, party_id)
        channel_infos.append(info)
    return channel_infos


MESSAGE_MAX_SIZE = 200000


def _partition_send(kvs, name, tag, total_size, partitions, mq_names, mq):
    LOGGER.debug(
        f"_partition_send, total_size:{total_size}, partitions:{partitions}, mq_names:{mq_names}, mq:{mq}.")
    channel_infos = _get_channels(mq_names=mq_names, mq=mq)
    data = []
    lines = 0
    for k, v in kvs:
        el = {'k': p_dumps(k).hex(), 'v': p_dumps(v).hex()}
        data.append(el)
        lines = lines + 1
        if lines > MESSAGE_MAX_SIZE:
            _send_kv(name=name, tag=tag, data=data, channel_infos=channel_infos,
                     total_size=total_size, partitions=partitions)
            lines = 0
            data.clear()
    _send_kv(name=name, tag=tag, data=data, channel_infos=channel_infos, total_size=total_size,
             partitions=partitions)
    return data


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
                data_iter = ((p_loads(bytes.fromhex(el['k'])), p_loads(bytes.fromhex(el['v']))) for el in data)
                sc = SparkContext.getOrCreate()
                if obj:
                    rdd = sc.parallelize(data_iter, properties.headers["partitions"])
                    obj = obj.union(rdd)
                    LOGGER.debug("before coalesce: federation get union partition %d, count: %d" % (
                        obj.getNumPartitions(), obj.count()))
                    obj = obj.coalesce(properties.headers["partitions"])
                    LOGGER.debug("end coalesce: federation get union partition %d, count: %d" % (
                        obj.getNumPartitions(), obj.count()))
                else:
                    obj = sc.parallelize(data_iter, properties.headers["partitions"]).persist(
                        get_storage_level())
                if count == properties.headers["total_size"]:
                    channel_info.basic_ack(delivery_tag=method.delivery_tag)
                    break

            channel_info.basic_ack(delivery_tag=method.delivery_tag)
    # return any pending messages
    channel_info.cancel()
    return obj
