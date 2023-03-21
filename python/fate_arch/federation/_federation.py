#
#  Copyright 2022 The FATE Authors. All Rights Reserved.
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
import sys
import typing
from pickle import dumps as p_dumps, loads as p_loads

from fate_arch.abc import CTableABC
from fate_arch.abc import FederationABC, GarbageCollectionABC
from fate_arch.common import Party
from fate_arch.common.log import getLogger
from fate_arch.federation import FederationDataType
from fate_arch.federation._datastream import Datastream
from fate_arch.session import computing_session

LOGGER = getLogger()

NAME_DTYPE_TAG = "<dtype>"
_SPLIT_ = "^"


def _get_splits(obj, max_message_size):
    obj_bytes = p_dumps(obj, protocol=4)
    byte_size = len(obj_bytes)
    num_slice = (byte_size - 1) // max_message_size + 1
    if num_slice <= 1:
        return obj, num_slice
    else:
        _max_size = max_message_size
        kv = [(i, obj_bytes[slice(i * _max_size, (i + 1) * _max_size)]) for i in range(num_slice)]
        return kv, num_slice


class FederationBase(FederationABC):
    @staticmethod
    def from_conf(
            federation_session_id: str,
            party: Party,
            runtime_conf: dict,
            **kwargs
    ):
        raise NotImplementedError()

    def __init__(
            self,
            session_id,
            party: Party,
            mq,
            max_message_size,
            conf=None
    ):
        self._session_id = session_id
        self._party = party
        self._mq = mq
        self._topic_map = {}
        self._channels_map = {}
        self._name_dtype_map = {}
        self._message_cache = {}
        self._max_message_size = max_message_size
        self._conf = conf

    def __getstate__(self):
        pass

    @property
    def session_id(self) -> str:
        return self._session_id

    def destroy(self, parties):
        raise NotImplementedError()

    def get(
            self, name: str, tag: str, parties: typing.List[Party], gc: GarbageCollectionABC
    ) -> typing.List:
        log_str = f"[federation.get](name={name}, tag={tag}, parties={parties})"
        LOGGER.debug(f"[{log_str}]start to get")

        _name_dtype_keys = [
            _SPLIT_.join([party.role, party.party_id, name, tag, "get"])
            for party in parties
        ]

        if _name_dtype_keys[0] not in self._name_dtype_map:
            party_topic_infos = self._get_party_topic_infos(parties, dtype=NAME_DTYPE_TAG)
            channel_infos = self._get_channels(party_topic_infos=party_topic_infos)
            rtn_dtype = []
            for i, info in enumerate(channel_infos):
                obj = self._receive_obj(
                    info, name, tag=_SPLIT_.join([tag, NAME_DTYPE_TAG])
                )
                rtn_dtype.append(obj)
                LOGGER.debug(
                    f"[federation.get] _name_dtype_keys: {_name_dtype_keys}, dtype: {obj}"
                )

            for k in _name_dtype_keys:
                if k not in self._name_dtype_map:
                    self._name_dtype_map[k] = rtn_dtype[0]

        rtn_dtype = self._name_dtype_map[_name_dtype_keys[0]]

        rtn = []
        dtype = rtn_dtype.get("dtype", None)
        partitions = rtn_dtype.get("partitions", None)

        if dtype == FederationDataType.TABLE or dtype == FederationDataType.SPLIT_OBJECT:
            party_topic_infos = self._get_party_topic_infos(parties, name, partitions=partitions)
            for i in range(len(party_topic_infos)):
                party = parties[i]
                role = party.role
                party_id = party.party_id
                topic_infos = party_topic_infos[i]
                receive_func = self._get_partition_receive_func(
                    name=name,
                    tag=tag,
                    src_party_id=self._party.party_id,
                    src_role=self._party.role,
                    dst_party_id=party_id,
                    dst_role=role,
                    topic_infos=topic_infos,
                    mq=self._mq,
                    conf=self._conf
                )

                table = computing_session.parallelize(range(partitions), partitions, include_key=False)
                table = table.mapPartitionsWithIndex(receive_func)

                # add gc
                gc.add_gc_action(tag, table, "__del__", {})

                LOGGER.debug(
                    f"[{log_str}]received table({i + 1}/{len(parties)}), party: {parties[i]} "
                )
                if dtype == FederationDataType.TABLE:
                    rtn.append(table)
                else:
                    obj_bytes = b''.join(map(lambda t: t[1], sorted(table.collect(), key=lambda x: x[0])))
                    obj = p_loads(obj_bytes)
                    rtn.append(obj)
        else:
            party_topic_infos = self._get_party_topic_infos(parties, name)
            channel_infos = self._get_channels(party_topic_infos=party_topic_infos)
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
        log_str = f"[federation.remote](name={name}, tag={tag}, parties={parties})"

        _name_dtype_keys = [
            _SPLIT_.join([party.role, party.party_id, name, tag, "remote"])
            for party in parties
        ]

        if _name_dtype_keys[0] not in self._name_dtype_map:
            party_topic_infos = self._get_party_topic_infos(parties, dtype=NAME_DTYPE_TAG)
            channel_infos = self._get_channels(party_topic_infos=party_topic_infos)

            if not isinstance(v, CTableABC):
                v, num_slice = _get_splits(v, self._max_message_size)
                if num_slice > 1:
                    v = computing_session.parallelize(data=v, partition=1, include_key=True)
                    body = {"dtype": FederationDataType.SPLIT_OBJECT, "partitions": v.partitions}
                else:
                    body = {"dtype": FederationDataType.OBJECT}

            else:
                body = {"dtype": FederationDataType.TABLE, "partitions": v.partitions}

            LOGGER.debug(
                f"[federation.remote] _name_dtype_keys: {_name_dtype_keys}, dtype: {body}"
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

        if isinstance(v, CTableABC):
            total_size = v.count()
            partitions = v.partitions
            LOGGER.debug(
                f"[{log_str}]start to remote table, total_size={total_size}, partitions={partitions}"
            )

            party_topic_infos = self._get_party_topic_infos(parties, name, partitions=partitions)
            # add gc
            gc.add_gc_action(tag, v, "__del__", {})

            send_func = self._get_partition_send_func(
                name=name,
                tag=tag,
                partitions=partitions,
                party_topic_infos=party_topic_infos,
                src_party_id=self._party.party_id,
                src_role=self._party.role,
                mq=self._mq,
                max_message_size=self._max_message_size,
                conf=self._conf
            )
            # noinspection PyProtectedMember
            v.mapPartitionsWithIndex(send_func)
        else:
            LOGGER.debug(f"[{log_str}]start to remote obj")
            party_topic_infos = self._get_party_topic_infos(parties, name)
            channel_infos = self._get_channels(party_topic_infos=party_topic_infos)
            self._send_obj(
                name=name, tag=tag, data=p_dumps(v), channel_infos=channel_infos
            )

        LOGGER.debug(f"[{log_str}]finish to remote")

    def _get_party_topic_infos(
            self, parties: typing.List[Party], name=None, partitions=None, dtype=None
    ) -> typing.List:
        topic_infos = [
            self._get_or_create_topic(party, name, partitions, dtype)
            for party in parties
        ]
        return topic_infos

    def _maybe_create_topic_and_replication(self, party, topic_suffix):
        # gen names
        raise NotImplementedError()

    def _get_or_create_topic(
            self, party: Party, name=None, partitions=None, dtype=None
    ) -> typing.Tuple:
        topic_key_list = []
        topic_infos = []

        if dtype is not None:
            topic_key = _SPLIT_.join(
                [party.role, party.party_id, dtype, dtype])
            topic_key_list.append(topic_key)
        else:
            if partitions is not None:
                for i in range(partitions):
                    topic_key = _SPLIT_.join(
                        [party.role, party.party_id, name, str(i)])
                    topic_key_list.append(topic_key)
            elif name is not None:
                topic_key = _SPLIT_.join([party.role, party.party_id, name])
                topic_key_list.append(topic_key)
            else:
                topic_key = _SPLIT_.join([party.role, party.party_id])
                topic_key_list.append(topic_key)

        for topic_key in topic_key_list:
            if topic_key not in self._topic_map:
                topic_key_splits = topic_key.split(_SPLIT_)
                topic_suffix = "-".join(topic_key_splits[2:])
                topic_pair = self._maybe_create_topic_and_replication(party, topic_suffix)
                self._topic_map[topic_key] = topic_pair

            topic_pair = self._topic_map[topic_key]
            topic_infos.append((topic_key, topic_pair))

        return topic_infos

    def _get_channel(
            self, topic_pair, src_party_id, src_role, dst_party_id, dst_role, mq=None, conf: dict = None):
        raise NotImplementedError()

    def _get_channels(self, party_topic_infos):
        channel_infos = []
        for e in party_topic_infos:
            for topic_key, topic_pair in e:
                topic_key_splits = topic_key.split(_SPLIT_)
                role = topic_key_splits[0]
                party_id = topic_key_splits[1]
                info = self._channels_map.get(topic_key)

                if info is None:
                    info = self._get_channel(
                        topic_pair=topic_pair,
                        src_party_id=self._party.party_id,
                        src_role=self._party.role,
                        dst_party_id=party_id,
                        dst_role=role,
                        mq=self._mq,
                        conf=self._conf
                    )

                    self._channels_map[topic_key] = info
                channel_infos.append(info)
        return channel_infos

    def _get_channels_index(self, index, party_topic_infos, src_party_id, src_role, mq=None, conf: dict = None):
        channel_infos = []
        for e in party_topic_infos:
            # select specified topic_info for a party
            topic_key, topic_pair = e[index]
            topic_key_splits = topic_key.split(_SPLIT_)
            role = topic_key_splits[0]
            party_id = topic_key_splits[1]
            info = self._get_channel(
                topic_pair=topic_pair,
                src_party_id=src_party_id,
                src_role=src_role,
                dst_party_id=party_id,
                dst_role=role,
                mq=mq,
                conf=conf
            )
            channel_infos.append(info)
        return channel_infos

    def _send_obj(self, name, tag, data, channel_infos):
        for info in channel_infos:
            properties = {
                "content_type": "text/plain",
                "app_id": info._dst_party_id,
                "message_id": name,
                "correlation_id": tag
            }
            LOGGER.debug(f"[federation._send_obj]properties:{properties}.")
            info.produce(body=data, properties=properties)

    def _send_kv(
            self, name, tag, data, channel_infos, partition_size, partitions, message_key
    ):
        headers = json.dumps(
            {
                "partition_size": partition_size,
                "partitions": partitions,
                "message_key": message_key
            }
        )
        for info in channel_infos:
            properties = {
                "content_type": "application/json",
                "app_id": info._dst_party_id,
                "message_id": name,
                "correlation_id": tag,
                "headers": headers
            }
            print(f"[federation._send_kv]info: {info}, properties: {properties}.")
            info.produce(body=data, properties=properties)

    def _get_partition_send_func(
            self,
            name,
            tag,
            partitions,
            party_topic_infos,
            src_party_id,
            src_role,
            mq,
            max_message_size,
            conf: dict,
    ):
        def _fn(index, kvs):
            return self._partition_send(
                index=index,
                kvs=kvs,
                name=name,
                tag=tag,
                partitions=partitions,
                party_topic_infos=party_topic_infos,
                src_party_id=src_party_id,
                src_role=src_role,
                mq=mq,
                max_message_size=max_message_size,
                conf=conf,
            )

        return _fn

    def _partition_send(
            self,
            index,
            kvs,
            name,
            tag,
            partitions,
            party_topic_infos,
            src_party_id,
            src_role,
            mq,
            max_message_size,
            conf: dict,
    ):
        channel_infos = self._get_channels_index(
            index=index, party_topic_infos=party_topic_infos, src_party_id=src_party_id, src_role=src_role, mq=mq,
            conf=conf
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
                    >= max_message_size
            ):
                print(
                    f"[federation._partition_send]The size of message is: {datastream.get_size()}"
                )
                message_key_idx += 1
                message_key = base_message_key + "_" + str(message_key_idx)
                self._send_kv(
                    name=name,
                    tag=tag,
                    data=datastream.get_data().encode(),
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
            data=datastream.get_data().encode(),
            channel_infos=channel_infos,
            partition_size=count,
            partitions=partitions,
            message_key=message_key,
        )

        return [(index, 1)]

    def _get_message_cache_key(self, name, tag, party_id, role):
        cache_key = _SPLIT_.join([name, tag, str(party_id), role])
        return cache_key

    def _get_consume_message(self, channel_info):
        raise NotImplementedError()

    def _consume_ack(self, channel_info, id):
        raise NotImplementedError()

    def _query_receive_topic(self, channel_info):
        return channel_info

    def _receive_obj(self, channel_info, name, tag):
        party_id = channel_info._dst_party_id
        role = channel_info._dst_role

        wish_cache_key = self._get_message_cache_key(name, tag, party_id, role)

        if wish_cache_key in self._message_cache:
            recv_obj = self._message_cache[wish_cache_key]
            del self._message_cache[wish_cache_key]
            return recv_obj

        channel_info = self._query_receive_topic(channel_info)

        for id, properties, body in self._get_consume_message(channel_info):
            LOGGER.debug(
                f"[federation._receive_obj] properties: {properties}"
            )
            if properties["message_id"] != name or properties["correlation_id"] != tag:
                # todo: fix this
                LOGGER.warning(
                    f"[federation._receive_obj] require {name}.{tag}, got {properties['message_id']}.{properties['correlation_id']}"
                )

            cache_key = self._get_message_cache_key(
                properties["message_id"], properties["correlation_id"], party_id, role
            )
            # object
            if properties["content_type"] == "text/plain":
                recv_obj = p_loads(body)
                self._consume_ack(channel_info, id)
                LOGGER.debug(
                    f"[federation._receive_obj] cache_key: {cache_key}, wish_cache_key: {wish_cache_key}"
                )
                if cache_key == wish_cache_key:
                    channel_info.cancel()
                    return recv_obj
                else:
                    self._message_cache[cache_key] = recv_obj
            else:
                raise ValueError(
                    f"[federation._receive_obj] properties.content_type is {properties['content_type']}, but must be text/plain"
                )

    def _get_partition_receive_func(
            self, name, tag, src_party_id, src_role, dst_party_id, dst_role, topic_infos, mq, conf: dict
    ):
        def _fn(index, kvs):
            return self._partition_receive(
                index=index,
                kvs=kvs,
                name=name,
                tag=tag,
                src_party_id=src_party_id,
                src_role=src_role,
                dst_party_id=dst_party_id,
                dst_role=dst_role,
                topic_infos=topic_infos,
                mq=mq,
                conf=conf,
            )

        return _fn

    def _partition_receive(
            self,
            index,
            kvs,
            name,
            tag,
            src_party_id,
            src_role,
            dst_party_id,
            dst_role,
            topic_infos,
            mq,
            conf: dict,
    ):
        topic_pair = topic_infos[index][1]
        channel_info = self._get_channel(topic_pair=topic_pair,
                                         src_party_id=src_party_id,
                                         src_role=src_role,
                                         dst_party_id=dst_party_id,
                                         dst_role=dst_role,
                                         mq=mq,
                                         conf=conf)

        message_key_cache = set()
        count = 0
        partition_size = -1
        all_data = []

        channel_info = self._query_receive_topic(channel_info)

        while True:
            try:
                for id, properties, body in self._get_consume_message(channel_info):
                    print(
                        f"[federation._partition_receive] properties: {properties}."
                    )
                    if properties["message_id"] != name or properties["correlation_id"] != tag:
                        # todo: fix this
                        self._consume_ack(channel_info, id)
                        print(
                            f"[federation._partition_receive]: require {name}.{tag}, got {properties['message_id']}.{properties['correlation_id']}"
                        )
                        continue

                    if properties["content_type"] == "application/json":
                        header = json.loads(properties["headers"])
                        message_key = header["message_key"]
                        if message_key in message_key_cache:
                            print(
                                f"[federation._partition_receive] message_key : {message_key} is duplicated"
                            )
                            self._consume_ack(channel_info, id)
                            continue

                        message_key_cache.add(message_key)

                        if header["partition_size"] >= 0:
                            partition_size = header["partition_size"]

                        data = json.loads(body.decode())
                        data_iter = (
                            (p_loads(bytes.fromhex(el["k"])), p_loads(bytes.fromhex(el["v"])))
                            for el in data
                        )
                        count += len(data)
                        print(f"[federation._partition_receive] count: {count}")
                        all_data.extend(data_iter)
                        self._consume_ack(channel_info, id)

                        if count == partition_size:
                            channel_info.cancel()
                            return all_data
                    else:
                        ValueError(
                            f"[federation._partition_receive]properties.content_type is {properties['content_type']}, but must be application/json"
                        )

            except Exception as e:
                LOGGER.error(
                    f"[federation._partition_receive]catch exception {e}, while receiving {name}.{tag}"
                )
                # avoid hang on consume()
                if count == partition_size:
                    channel_info.cancel()
                    return all_data
                else:
                    raise e
