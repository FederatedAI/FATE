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

import io
import json
import sys
import time
import typing
from pickle import dumps as p_dumps, loads as p_loads

import pika

# noinspection PyPackageRequirements
from pyspark import SparkContext

from fate_arch.common import file_utils, string_utils
from fate_arch.abc import FederationABC, GarbageCollectionABC
from fate_arch.common import Party
from fate_arch.common.log import getLogger
from fate_arch.computing.spark import Table
from fate_arch.computing.spark._materialize import materialize
from fate_arch.federation.rabbitmq._mq_channel import MQChannel
from fate_arch.federation.rabbitmq._rabbit_manager import RabbitManager
from fate_arch.federation._datastream import Datastream


LOGGER = getLogger()

# default message max size in bytes = 1MB
DEFAULT_MESSAGE_MAX_SIZE = 1048576
NAME_DTYPE_TAG = "<dtype>"
_SPLIT_ = "^"


class FederationDataType(object):
    OBJECT = "obj"
    TABLE = "Table"


class MQ(object):
    def __init__(self, host, port, union_name, policy_id, route_table):
        self.host = host
        self.port = port
        self.union_name = union_name
        self.policy_id = policy_id
        self.route_table = route_table

    def __str__(self):
        return (
            f"MQ(host={self.host}, port={self.port}, union_name={self.union_name}, "
            f"policy_id={self.policy_id}, route_table={self.route_table})"
        )

    def __repr__(self):
        return self.__str__()


class _QueueNames(object):
    def __init__(self, vhost, send, receive):
        self.vhost = vhost
        # self.union = union
        self.send = send
        self.receive = receive


_remote_history = set()


def _remote_tag_not_duplicate(name, tag, parties):
    for party in parties:
        if (name, tag, party) in _remote_history:
            return False
        _remote_history.add((name, tag, party))
    return True


_get_history = set()


def _get_tag_not_duplicate(name, tag, party):
    if (name, tag, party) in _get_history:
        return False
    _get_history.add((name, tag, party))
    return True


class Federation(FederationABC):
    @staticmethod
    def from_conf(
        federation_session_id: str,
        party: Party,
        runtime_conf: dict,
        rabbitmq_config: dict,
    ):
        LOGGER.debug(f"rabbitmq_config: {rabbitmq_config}")
        host = rabbitmq_config.get("host")
        port = rabbitmq_config.get("port")
        mng_port = rabbitmq_config.get("mng_port")
        base_user = rabbitmq_config.get("user")
        base_password = rabbitmq_config.get("password")

        """
        federation_info = runtime_conf.get("job_parameters", {}).get(
            "federation_info", {}
        )
        union_name = federation_info.get("union_name")
        policy_id = federation_info.get("policy_id")
        """

        # union_name = string_utils.random_string(4)
        # policy_id = string_utils.random_string(10)

        union_name = federation_session_id
        policy_id = federation_session_id

        rabbitmq_run = runtime_conf.get("job_parameters", {}).get("rabbitmq_run", {})
        LOGGER.debug(f"rabbitmq_run: {rabbitmq_run}")
        max_message_size = rabbitmq_run.get(
            "max_message_size", DEFAULT_MESSAGE_MAX_SIZE
        )
        LOGGER.debug(f"set max message size to {max_message_size} Bytes")

        rabbit_manager = RabbitManager(
            base_user, base_password, f"{host}:{mng_port}", rabbitmq_run
        )
        rabbit_manager.create_user(union_name, policy_id)
        route_table_path = rabbitmq_config.get("route_table")
        if route_table_path is None:
            route_table_path = "conf/rabbitmq_route_table.yaml"
        route_table = file_utils.load_yaml_conf(conf_path=route_table_path)
        mq = MQ(host, port, union_name, policy_id, route_table)
        return Federation(
            federation_session_id, party, mq, rabbit_manager, max_message_size
        )

    def __init__(
        self,
        session_id,
        party: Party,
        mq: MQ,
        rabbit_manager: RabbitManager,
        max_message_size,
    ):
        self._session_id = session_id
        self._party = party
        self._mq = mq
        self._rabbit_manager = rabbit_manager

        self._queue_map: typing.MutableMapping[_QueueKey, _QueueNames] = {}
        self._channels_map: typing.MutableMapping[_QueueKey, MQChannel] = {}
        self._vhost_set = set()
        self._name_dtype_map = {}
        self._message_cache = {}
        self._max_message_size = max_message_size

    def __getstate__(self):
        pass

    def get(
        self, name: str, tag: str, parties: typing.List[Party], gc: GarbageCollectionABC
    ) -> typing.List:
        log_str = f"[rabbitmq.get](name={name}, tag={tag}, parties={parties})"
        LOGGER.debug(f"[{log_str}]start to get")

        # for party in parties:
        #     if not _get_tag_not_duplicate(name, tag, party):
        #         raise ValueError(f"[{log_str}]get from {party} with duplicate tag")

        _name_dtype_keys = [
            _SPLIT_.join([party.role, party.party_id, name, tag, "get"])
            for party in parties
        ]

        if _name_dtype_keys[0] not in self._name_dtype_map:
            mq_names = self._get_mq_names(parties, dtype=NAME_DTYPE_TAG)
            channel_infos = self._get_channels(mq_names=mq_names)
            rtn_dtype = []
            for i, info in enumerate(channel_infos):
                obj = self._receive_obj(
                    info, name, tag=_SPLIT_.join([tag, NAME_DTYPE_TAG])
                )
                rtn_dtype.append(obj)
                LOGGER.debug(
                    f"[rabbitmq.get] _name_dtype_keys: {_name_dtype_keys}, dtype: {obj}"
                )

            for k in _name_dtype_keys:
                if k not in self._name_dtype_map:
                    self._name_dtype_map[k] = rtn_dtype[0]

        rtn_dtype = self._name_dtype_map[_name_dtype_keys[0]]

        rtn = []
        dtype = rtn_dtype.get("dtype", None)
        partitions = rtn_dtype.get("partitions", None)

        if dtype == FederationDataType.TABLE:
            mq_names = self._get_mq_names(parties, name, partitions=partitions)
            for i in range(len(mq_names)):
                party = parties[i]
                role = party.role
                party_id = party.party_id
                party_mq_names = mq_names[i]
                receive_func = self._get_partition_receive_func(
                    name,
                    tag,
                    party_id,
                    role,
                    party_mq_names,
                    mq=self._mq,
                    connection_conf=self._rabbit_manager.runtime_config.get(
                        "connection", {}
                    ),
                )

                sc = SparkContext.getOrCreate()
                rdd = sc.parallelize(range(partitions), partitions)
                rdd = rdd.mapPartitionsWithIndex(receive_func)
                rdd = materialize(rdd)
                table = Table(rdd)
                rtn.append(table)
                # add gc
                gc.add_gc_action(tag, table, "__del__", {})

                LOGGER.debug(
                    f"[{log_str}]received rdd({i + 1}/{len(parties)}), party: {parties[i]} "
                )
        else:
            mq_names = self._get_mq_names(parties, name)
            channel_infos = self._get_channels(mq_names=mq_names)
            for i, info in enumerate(channel_infos):
                obj = self._receive_obj(info, name, tag)
                LOGGER.debug(
                    f"[{log_str}]received obj({i + 1}/{len(parties)}), party: {parties[i]} "
                )
                rtn.append(obj)

        LOGGER.debug(f"[{log_str}]finish to get")
        return rtn

    def remote(
        self,
        v,
        name: str,
        tag: str,
        parties: typing.List[Party],
        gc: GarbageCollectionABC,
    ) -> typing.NoReturn:
        log_str = f"[rabbitmq.remote](name={name}, tag={tag}, parties={parties})"

        # if not _remote_tag_not_duplicate(name, tag, parties):
        #     raise ValueError(f"[{log_str}]remote to {parties} with duplicate tag")

        _name_dtype_keys = [
            _SPLIT_.join([party.role, party.party_id, name, tag, "remote"])
            for party in parties
        ]

        if _name_dtype_keys[0] not in self._name_dtype_map:
            mq_names = self._get_mq_names(parties, dtype=NAME_DTYPE_TAG)
            channel_infos = self._get_channels(mq_names=mq_names)
            if isinstance(v, Table):
                body = {"dtype": FederationDataType.TABLE, "partitions": v.partitions}
            else:
                body = {"dtype": FederationDataType.OBJECT}

            LOGGER.debug(
                f"[rabbitmq.remote] _name_dtype_keys: {_name_dtype_keys}, dtype: {body}"
            )
            self._send_obj(
                name=name,
                tag=_SPLIT_.join([tag, NAME_DTYPE_TAG]),
                data=p_dumps(body),
                channel_infos=channel_infos,
            )

            for k in _name_dtype_keys:
                if k not in self._name_dtype_map:
                    self._name_dtype_map[k] = body

        if isinstance(v, Table):
            total_size = v.count()
            partitions = v.partitions
            LOGGER.debug(
                f"[{log_str}]start to remote RDD, total_size={total_size}, partitions={partitions}"
            )

            mq_names = self._get_mq_names(parties, name, partitions=partitions)
            # add gc
            gc.add_gc_action(tag, v, "__del__", {})

            send_func = self._get_partition_send_func(
                name,
                tag,
                partitions,
                mq_names,
                mq=self._mq,
                maximun_message_size=self._max_message_size,
                connection_conf=self._rabbit_manager.runtime_config.get(
                    "connection", {}
                ),
            )
            # noinspection PyProtectedMember
            v._rdd.mapPartitionsWithIndex(send_func).count()
        else:
            LOGGER.debug(f"[{log_str}]start to remote obj")
            mq_names = self._get_mq_names(parties, name)
            channel_infos = self._get_channels(mq_names=mq_names)
            self._send_obj(
                name=name, tag=tag, data=p_dumps(v), channel_infos=channel_infos
            )

        LOGGER.debug(f"[{log_str}]finish to remote")

    def cleanup(self, parties):
        LOGGER.debug("[rabbitmq.cleanup]start to cleanup...")
        for party in parties:
            vhost = self._get_vhost(party)
            LOGGER.debug(f"[rabbitmq.cleanup]start to cleanup vhost {vhost}...")
            self._rabbit_manager.clean(vhost)
            LOGGER.debug(f"[rabbitmq.cleanup]cleanup vhost {vhost} done")
        if self._mq.union_name:
            LOGGER.debug(f"[rabbitmq.cleanup]clean user {self._mq.union_name}.")
            self._rabbit_manager.delete_user(user=self._mq.union_name)

    def _get_vhost(self, party):
        low, high = (
            (self._party, party) if self._party < party else (party, self._party)
        )
        vhost = (
            f"{self._session_id}-{low.role}-{low.party_id}-{high.role}-{high.party_id}"
        )
        return vhost

    def _get_mq_names(
        self, parties: typing.List[Party], name=None, partitions=None, dtype=None
    ) -> typing.List:
        mq_names = [
            self._get_or_create_queue(party, name, partitions, dtype)
            for party in parties
        ]
        return mq_names

    def _get_or_create_queue(
        self, party: Party, name=None, partitions=None, dtype=None
    ) -> typing.Tuple:
        queue_key_list = []
        queue_infos = []

        if dtype is not None:
            queue_key = _SPLIT_.join([party.role, party.party_id, dtype, dtype])
            queue_key_list.append(queue_key)
        else:
            if partitions is not None:
                for i in range(partitions):
                    queue_key = _SPLIT_.join([party.role, party.party_id, name, str(i)])
                    queue_key_list.append(queue_key)
            elif name is not None:
                queue_key = _SPLIT_.join([party.role, party.party_id, name])
                queue_key_list.append(queue_key)
            else:
                queue_key = _SPLIT_.join([party.role, party.party_id])
                queue_key_list.append(queue_key)

        for queue_key in queue_key_list:
            if queue_key not in self._queue_map:
                LOGGER.debug(
                    f"[rabbitmq.get_or_create_queue]queue: {queue_key} for party:{party} not found, start to create"
                )
                # gen names
                vhost_name = self._get_vhost(party)

                queue_key_splits = queue_key.split(_SPLIT_)
                queue_suffix = "-".join(queue_key_splits[2:])
                send_queue_name = f"send-{self._session_id}-{self._party.role}-{self._party.party_id}-{party.role}-{party.party_id}-{queue_suffix}"
                receive_queue_name = f"receive-{self._session_id}-{party.role}-{party.party_id}-{self._party.role}-{self._party.party_id}-{queue_suffix}"

                queue_names = _QueueNames(
                    vhost_name, send_queue_name, receive_queue_name
                )

                # initial vhost
                if queue_names.vhost not in self._vhost_set:
                    self._rabbit_manager.create_vhost(queue_names.vhost)
                    self._rabbit_manager.add_user_to_vhost(
                        self._mq.union_name, queue_names.vhost
                    )
                    self._vhost_set.add(queue_names.vhost)

                # initial send queue, the name is send-${vhost}
                self._rabbit_manager.create_queue(queue_names.vhost, queue_names.send)

                # initial receive queue, the name is receive-${vhost}
                self._rabbit_manager.create_queue(
                    queue_names.vhost, queue_names.receive
                )

                upstream_uri = self._upstream_uri(party_id=party.party_id)
                self._rabbit_manager.federate_queue(
                    upstream_host=upstream_uri,
                    vhost=queue_names.vhost,
                    send_queue_name=queue_names.send,
                    receive_queue_name=queue_names.receive,
                )

                self._queue_map[queue_key] = queue_names
                # TODO: check federated queue status
                LOGGER.debug(
                    f"[rabbitmq.get_or_create_queue]queue for queue_key: {queue_key}, party:{party} created"
                )

            queue_names = self._queue_map[queue_key]
            queue_infos.append((queue_key, queue_names))

        return queue_infos

    def _upstream_uri(self, party_id):
        host = self._mq.route_table.get(int(party_id)).get("host")
        port = self._mq.route_table.get(int(party_id)).get("port")
        upstream_uri = (
            f"amqp://{self._mq.union_name}:{self._mq.policy_id}@{host}:{port}"
        )
        return upstream_uri

    def _get_channel(
        self, mq, queue_names: _QueueNames, party_id, role, connection_conf: dict
    ):
        return MQChannel(
            host=mq.host,
            port=mq.port,
            user=mq.union_name,
            password=mq.policy_id,
            vhost=queue_names.vhost,
            send_queue_name=queue_names.send,
            receive_queue_name=queue_names.receive,
            party_id=party_id,
            role=role,
            extra_args=connection_conf,
        )

    def _get_channels(self, mq_names):
        channel_infos = []
        for e in mq_names:
            for queue_key, queue_names in e:
                queue_key_splits = queue_key.split(_SPLIT_)
                role = queue_key_splits[0]
                party_id = queue_key_splits[1]
                info = self._channels_map.get(queue_key)
                if info is None:
                    info = self._get_channel(
                        self._mq,
                        queue_names,
                        party_id=party_id,
                        role=role,
                        connection_conf=self._rabbit_manager.runtime_config.get(
                            "connection", {}
                        ),
                    )
                    self._channels_map[queue_key] = info
                channel_infos.append(info)
        return channel_infos

    # can't pickle _thread.lock objects
    def _get_channels_index(self, index, mq_names, mq, connection_conf: dict):
        channel_infos = []
        for e in mq_names:
            queue_key, queue_names = e[index]
            queue_key_splits = queue_key.split(_SPLIT_)
            role = queue_key_splits[0]
            party_id = queue_key_splits[1]
            info = self._get_channel(
                mq,
                queue_names,
                party_id=party_id,
                role=role,
                connection_conf=connection_conf,
            )
            channel_infos.append(info)
        return channel_infos

    def _send_obj(self, name, tag, data, channel_infos):
        for info in channel_infos:
            properties = pika.BasicProperties(
                content_type="text/plain",
                app_id=info.party_id,
                message_id=name,
                correlation_id=tag,
                delivery_mode=1,
            )
            LOGGER.debug(f"[rabbitmq._send_obj]properties:{properties}.")
            info.basic_publish(body=data, properties=properties)

    def _get_message_cache_key(self, name, tag, party_id, role):
        cache_key = _SPLIT_.join([name, tag, str(party_id), role])
        return cache_key

    def _receive_obj(self, channel_info, name, tag):
        party_id = channel_info._party_id
        role = channel_info._role
        wish_cache_key = self._get_message_cache_key(name, tag, party_id, role)

        if wish_cache_key in self._message_cache:
            return self._message_cache[wish_cache_key]

        for method, properties, body in channel_info.consume():
            LOGGER.debug(
                f"[rabbitmq._receive_obj] method: {method}, properties: {properties}."
            )
            if properties.message_id != name or properties.correlation_id != tag:
                # todo: fix this
                LOGGER.warning(
                    f"[rabbitmq._receive_obj] require {name}.{tag}, got {properties.message_id}.{properties.correlation_id}"
                )

            cache_key = self._get_message_cache_key(
                properties.message_id, properties.correlation_id, party_id, role
            )
            # object
            if properties.content_type == "text/plain":
                self._message_cache[cache_key] = p_loads(body)
                channel_info.basic_ack(delivery_tag=method.delivery_tag)
                if cache_key == wish_cache_key:
                    channel_info.cancel()
                    LOGGER.debug(
                        f"[rabbitmq._receive_obj] cache_key: {cache_key}, obj: {self._message_cache[cache_key]}"
                    )
                    return self._message_cache[cache_key]
            else:
                raise ValueError(
                    f"[rabbitmq._receive_obj] properties.content_type is {properties.content_type}, but must be text/plain"
                )

    def _send_kv(
        self, name, tag, data, channel_infos, partition_size, partitions, message_key
    ):
        headers = {
            "partition_size": partition_size,
            "partitions": partitions,
            "message_key": message_key,
        }
        for info in channel_infos:
            properties = pika.BasicProperties(
                content_type="application/json",
                app_id=info.party_id,
                message_id=name,
                correlation_id=tag,
                headers=headers,
                delivery_mode=1,
            )
            print(f"[rabbitmq._send_kv]info: {info}, properties: {properties}.")
            info.basic_publish(body=data, properties=properties)

    def _get_partition_send_func(
        self,
        name,
        tag,
        partitions,
        mq_names,
        mq,
        maximun_message_size,
        connection_conf: dict,
    ):
        def _fn(index, kvs):
            return self._partition_send(
                index,
                kvs,
                name,
                tag,
                partitions,
                mq_names,
                mq,
                maximun_message_size,
                connection_conf,
            )

        return _fn

    def _partition_send(
        self,
        index,
        kvs,
        name,
        tag,
        partitions,
        mq_names,
        mq,
        maximun_message_size,
        connection_conf: dict,
    ):
        channel_infos = self._get_channels_index(
            index=index, mq_names=mq_names, mq=mq, connection_conf=connection_conf
        )

        datastream = Datastream()
        base_message_key = str(index)
        message_key_idx = 0
        count = 0

        for k, v in kvs:
            count += 1
            el = {"k": p_dumps(k).hex(), "v": p_dumps(v).hex()}
            # roughly caculate the size of package to avoid serialization ;)
            if (
                datastream.get_size() + sys.getsizeof(el["k"]) + sys.getsizeof(el["v"])
                >= maximun_message_size
            ):
                print(
                    f"[rabbitmq._partition_send]The size of message is: {datastream.get_size()}"
                )
                message_key_idx += 1
                message_key = base_message_key + "_" + str(message_key_idx)
                self._send_kv(
                    name=name,
                    tag=tag,
                    data=datastream.get_data(),
                    channel_infos=channel_infos,
                    partition_size=-1,
                    partitions=partitions,
                    message_key=message_key,
                )
                datastream.clear()
            datastream.append(el)

        message_key_idx += 1
        message_key = _SPLIT_.join([base_message_key, str(message_key_idx)])

        self._send_kv(
            name=name,
            tag=tag,
            data=datastream.get_data(),
            channel_infos=channel_infos,
            partition_size=count,
            partitions=partitions,
            message_key=message_key,
        )

        return [1]

    def _get_partition_receive_func(
        self, name, tag, party_id, role, party_mq_names, mq, connection_conf: dict
    ):
        def _fn(index, kvs):
            return self._partition_receive(
                index,
                kvs,
                name,
                tag,
                party_id,
                role,
                party_mq_names,
                mq,
                connection_conf,
            )

        return _fn

    def _partition_receive(
        self,
        index,
        kvs,
        name,
        tag,
        party_id,
        role,
        party_mq_names,
        mq,
        connection_conf: dict,
    ):
        queue_names = party_mq_names[index][1]
        channel_info = self._get_channel(
            mq, queue_names, party_id, role, connection_conf
        )

        message_key_cache = set()
        count = 0
        partition_size = -1
        all_data = []

        while True:
            try:
                for method, properties, body in channel_info.consume():
                    print(
                        f"[rabbitmq._partition_receive] method: {method}, properties: {properties}."
                    )
                    if properties.message_id != name or properties.correlation_id != tag:
                        # todo: fix this
                        channel_info.basic_ack(delivery_tag=method.delivery_tag)
                        print(
                            f"[rabbitmq._partition_receive]: require {name}.{tag}, got {properties.message_id}.{properties.correlation_id}"
                        )
                        continue

                    if properties.content_type == "application/json":
                        message_key = properties.headers["message_key"]
                        if message_key in message_key_cache:
                            print(
                                f"[rabbitmq._partition_receive] message_key : {message_key} is duplicated"
                            )
                            channel_info.basic_ack(delivery_tag=method.delivery_tag)
                            continue

                        message_key_cache.add(message_key)

                        if properties.headers["partition_size"] >= 0:
                            partition_size = properties.headers["partition_size"]

                        data = json.loads(body)
                        data_iter = (
                            (p_loads(bytes.fromhex(el["k"])), p_loads(bytes.fromhex(el["v"])))
                            for el in data
                        )
                        count += len(data)
                        print(f"[rabbitmq._partition_receive] count: {count}")
                        all_data.extend(data_iter)
                        channel_info.basic_ack(delivery_tag=method.delivery_tag)

                        if count == partition_size:
                            channel_info.cancel()
                            return all_data
                    else:
                        ValueError(
                            f"[rabbitmq._partition_receive]properties.content_type is {properties.content_type}, but must be application/json"
                        )

            except Exception as e:
                LOGGER.error(
                    f"[rabbitmq._partition_receive]catch exception {e}, while receiving {name}.{tag}"
                )
                # avoid hang on consume()
                if count == partition_size:
                    channel_info.cancel()
                    return all_data
                else:
                    raise e
