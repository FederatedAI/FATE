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
import logging
import sys
import typing
from typing import List

from fate.arch.computing.api import KVTableContext
from fate.arch.federation.api import Federation, PartyMeta, TableMeta
from ._datastream import Datastream
from ._parties import Party

LOGGER = logging.getLogger(__name__)

_SPLIT_ = "^"


class MessageQueueBasedFederation(Federation):
    def __init__(
        self,
        session_id,
        computing_session: KVTableContext,
        party: PartyMeta,
        parties: typing.List[PartyMeta],
        mq,
        max_message_size,
        conf=None,
        default_partition_num=None,
    ):
        self._mq = mq
        self._topic_map = {}
        self._channels_map = {}
        self._message_cache = {}
        self._max_message_size = max_message_size
        if self._max_message_size is None:
            self._max_message_size = self.get_default_max_message_size()
        self._default_partition_num = default_partition_num
        self._conf = conf
        self.computing_session = computing_session

        super().__init__(session_id, party, parties)

        # TODO: remove this
        self._party = Party(party[0], party[1])

    def _get_channel(
        self,
        topic_pair,
        src_party_id,
        src_role,
        dst_party_id,
        dst_role,
        mq=None,
        conf: dict = None,
    ):
        raise NotImplementedError()

    def _maybe_create_topic_and_replication(self, party, topic_suffix):
        # gen names
        raise NotImplementedError()

    def _get_consume_message(self, channel_info):
        raise NotImplementedError()

    def _consume_ack(self, channel_info, id):
        return

    def get_default_max_message_size(self):
        if self._max_message_size is None:
            return super().get_default_max_message_size()
        else:
            return self._max_message_size

    def get_default_partition_num(self):
        if self._default_partition_num is None:
            return super().get_default_partition_num()
        else:
            return self._default_partition_num

    def _pull_bytes(self, name: str, tag: str, parties: typing.List[PartyMeta]) -> typing.List:
        _parties = [Party(role=p[0], party_id=p[1]) for p in parties]
        rtn = []
        party_topic_infos = self._get_party_topic_infos_by_name(_parties, name)
        channel_infos = self._get_channels(party_topic_infos=party_topic_infos)
        for i, info in enumerate(channel_infos):
            obj = self._receive_obj(info, name, tag)
            rtn.append(obj)

        return rtn

    def _push_bytes(
        self,
        v: bytes,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ):
        _parties = [Party(role=p[0], party_id=p[1]) for p in parties]
        party_topic_infos = self._get_party_topic_infos_by_name(_parties, name)
        channel_infos = self._get_channels(party_topic_infos=party_topic_infos)
        self._send_obj(name=name, tag=tag, data=v, channel_infos=channel_infos)

    def _pull_table(
        self, name: str, tag: str, parties: typing.List[PartyMeta], table_metas: List[TableMeta]
    ) -> typing.List:
        rtn = []
        for table_meta, party in zip(table_metas, parties):
            party = Party(role=party[0], party_id=party[1])
            party_topic_infos = self._get_party_topic_infos_by_name_and_partitions(
                [party], name, partitions=table_meta.num_partitions
            )[0]
            topic_infos = party_topic_infos
            receive_func = self._get_partition_receive_func(
                name=name,
                tag=tag,
                src_party_id=self.local_party[1],
                src_role=self.local_party[0],
                dst_party_id=party.party_id,
                dst_role=party.role,
                topic_infos=topic_infos,
                mq=self._mq,
                conf=self._conf,
            )
            table = self.computing_session.parallelize(
                range(table_meta.num_partitions),
                include_key=False,
                partition=table_meta.num_partitions,
            )
            table = table.mapPartitionsWithIndexNoSerdes(
                receive_func,
                output_key_serdes_type=table_meta.key_serdes_type,
                output_value_serdes_type=table_meta.value_serdes_type,
                output_partitioner_type=table_meta.partitioner_type,
            )
            rtn.append(table)
        return rtn

    def _push_table(self, table, name: str, tag: str, parties: typing.List[PartyMeta]):
        _parties = [Party(role=p[0], party_id=p[1]) for p in parties]
        party_topic_infos = self._get_party_topic_infos_by_name_and_partitions(
            _parties, name, partitions=table.num_partitions
        )
        send_func = self._get_partition_send_func(
            name=name,
            tag=tag,
            partitions=table.num_partitions,
            party_topic_infos=party_topic_infos,
            src_party_id=self.local_party[1],
            src_role=self.local_party[0],
            mq=self._mq,
            max_message_size=self._max_message_size,
            conf=self._conf,
        )
        # noinspection PyProtectedMember
        table.mapPartitionsWithIndexNoSerdes(
            send_func, output_key_serdes_type=0, output_value_serdes_type=0, output_partitioner_type=0
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    def __getstate__(self):
        pass

    def _get_party_topic_infos_by_name(self, parties: typing.List[Party], name: str):
        topic_infos = [[self._get_or_create_topic(party, "-".join([name]))] for party in parties]
        return topic_infos

    def _get_party_topic_infos_by_name_and_partitions(
        self, parties: typing.List[Party], name=None, partitions=None
    ) -> typing.List:
        topic_infos = [
            [self._get_or_create_topic(party, "-".join([name, str(i)])) for i in range(partitions)]
            for party in parties
        ]
        return topic_infos

    def _get_or_create_topic(self, party, topic_suffix) -> typing.Tuple[Party, str, typing.Any]:
        if (party, topic_suffix) not in self._topic_map:
            topic_pair = self._maybe_create_topic_and_replication(party, topic_suffix)
            self._topic_map[(party, topic_suffix)] = topic_pair

        topic_pair = self._topic_map[(party, topic_suffix)]
        return party, topic_suffix, topic_pair

    def _get_channels(self, party_topic_infos):
        channel_infos = []
        for e in party_topic_infos:
            for party, topic_suffix, topic_pair in e:
                info = self._channels_map.get((party, topic_suffix))

                if info is None:
                    info = self._get_channel(
                        topic_pair=topic_pair,
                        src_party_id=self.local_party[1],
                        src_role=self.local_party[0],
                        dst_party_id=party.party_id,
                        dst_role=party.role,
                        mq=self._mq,
                        conf=self._conf,
                    )

                    self._channels_map[(party, topic_suffix)] = info
                channel_infos.append(info)
        return channel_infos

    def _get_channels_index(
        self,
        index,
        party_topic_infos,
        src_party_id,
        src_role,
        mq=None,
        conf: dict = None,
    ):
        channel_infos = []
        for e in party_topic_infos:
            # select specified topic_info for a party
            party, topic_suffix, topic_pair = e[index]
            info = self._get_channel(
                topic_pair=topic_pair,
                src_party_id=src_party_id,
                src_role=src_role,
                dst_party_id=party.party_id,
                dst_role=party.role,
                mq=mq,
                conf=conf,
            )
            channel_infos.append(info)
        return channel_infos

    def _send_obj(self, name, tag, data, channel_infos):
        for info in channel_infos:
            properties = {
                "content_type": "text/plain",
                "app_id": info._dst_party_id,
                "message_id": name,
                "correlation_id": tag,
            }
            LOGGER.debug(f"[federation._send_obj]properties:{properties}.")
            info.produce(body=data, properties=properties)

    def _send_kv(self, name, tag, data, channel_infos, partition_size, partitions, message_key):
        headers = json.dumps(
            {
                "partition_size": partition_size,
                "partitions": partitions,
                "message_key": message_key,
            }
        )
        for info in channel_infos:
            properties = {
                "content_type": "application/json",
                "app_id": info._dst_party_id,
                "message_id": name,
                "correlation_id": tag,
                "headers": headers,
            }
            LOGGER.debug(f"[federation._send_kv]info: {info}, properties: {properties}.")
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
            index=index,
            party_topic_infos=party_topic_infos,
            src_party_id=src_party_id,
            src_role=src_role,
            mq=mq,
            conf=conf,
        )

        datastream = Datastream()
        base_message_key = str(index)
        message_key_idx = 0
        count = 0

        for k, v in kvs:
            count += 1
            el = {"k": k.hex(), "v": v.hex()}
            # roughly caculate the size of package to avoid serialization ;)
            if datastream.get_size() + sys.getsizeof(el["k"]) + sys.getsizeof(el["v"]) >= max_message_size:
                LOGGER.debug(f"[federation._partition_send]The size of message is: {datastream.get_size()}")
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

        return []

    def _receive_obj(self, channel_info, name, tag):
        party_id = channel_info._dst_party_id
        role = channel_info._dst_role

        wish_cache_key = _get_message_cache_key(name, tag, party_id, role)

        if wish_cache_key in self._message_cache:
            recv_bytes = self._message_cache[wish_cache_key]
            del self._message_cache[wish_cache_key]
            return recv_bytes

        # channel_info = self._query_receive_topic(channel_info)

        for _id, properties, body in self._get_consume_message(channel_info):
            LOGGER.debug(f"properties: {properties}")
            cache_key = _get_message_cache_key(properties["message_id"], properties["correlation_id"], party_id, role)
            # object
            if properties["content_type"] == "text/plain":
                recv_bytes = body
                self._consume_ack(channel_info, _id)
                LOGGER.debug(f"[federation._receive_obj] cache_key: {cache_key}, wish_cache_key: {wish_cache_key}")
                if cache_key == wish_cache_key:
                    channel_info.cancel()
                    return recv_bytes
                else:
                    self._message_cache[cache_key] = recv_bytes
            else:
                raise ValueError(
                    f"[federation._receive_obj] properties.content_type is {properties['content_type']}, but must be text/plain"
                )

    def _get_partition_receive_func(
        self,
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
        def _fn(index, _):
            return self._partition_receive(
                index=index,
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
        _, _, topic_pair = topic_infos[index]
        channel_info = self._get_channel(
            topic_pair=topic_pair,
            src_party_id=src_party_id,
            src_role=src_role,
            dst_party_id=dst_party_id,
            dst_role=dst_role,
            mq=mq,
            conf=conf,
        )

        message_key_cache = set()
        count = 0
        partition_size = -1
        all_data = []

        while True:
            try:
                for id, properties, body in self._get_consume_message(channel_info):
                    LOGGER.debug(f"[federation._partition_receive] properties: {properties}.")
                    if properties["message_id"] != name or properties["correlation_id"] != tag:
                        # todo: fix this
                        self._consume_ack(channel_info, id)
                        LOGGER.debug(
                            f"[federation._partition_receive]: require {name}.{tag}, got {properties['message_id']}.{properties['correlation_id']}"
                        )
                        continue

                    if properties["content_type"] == "application/json":
                        header = json.loads(properties["headers"])
                        message_key = header["message_key"]
                        if message_key in message_key_cache:
                            LOGGER.debug(f"[federation._partition_receive] message_key : {message_key} is duplicated")
                            self._consume_ack(channel_info, id)
                            continue

                        message_key_cache.add(message_key)

                        if header["partition_size"] >= 0:
                            partition_size = header["partition_size"]

                        data = json.loads(body.decode())
                        data_iter = (
                            (
                                bytes.fromhex(el["k"]),
                                bytes.fromhex(el["v"]),
                            )
                            for el in data
                        )
                        count += len(data)
                        LOGGER.debug(f"[federation._partition_receive] count: {count}")
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
                LOGGER.error(f"[federation._partition_receive]catch exception {e}, while receiving {name}.{tag}")
                # avoid hang on consume()
                if count == partition_size:
                    channel_info.cancel()
                    return all_data
                else:
                    raise e


def _get_message_cache_key(name: str, tag: str, party_id, role: str):
    cache_key = _SPLIT_.join([name, tag, str(party_id), role])
    return cache_key
