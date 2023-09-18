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

import json
import typing
from logging import getLogger

from fate.arch.abc import PartyMeta
from fate.arch.federation.osx import osx_pb2

from .._federation import FederationBase
from .._nretry import nretry
from ._mq_channel import MQChannel

LOGGER = getLogger(__name__)
# default message max size in bytes = 1MB
DEFAULT_MESSAGE_MAX_SIZE = 104857


class MQ(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def __str__(self):
        return f"MQ(host={self.host}, port={self.port}, type=osx)"

    def __repr__(self):
        return self.__str__()


class _TopicPair(object):
    def __init__(self, namespace, send, receive):
        self.namespace = namespace
        self.send = send
        self.receive = receive

    def __str__(self) -> str:
        return f"<_TopicPair namespace={self.namespace}, send={self.send}, receive={self.receive}>"


class OSXFederation(FederationBase):
    @staticmethod
    def from_conf(
        federation_session_id: str,
        computing_session,
        party: PartyMeta,
        parties: typing.List[PartyMeta],
        host: str,
        port: int,
        max_message_size: typing.Optional[int] = None,
    ):
        if max_message_size is None:
            max_message_size = DEFAULT_MESSAGE_MAX_SIZE
        mq = MQ(host, port)

        return OSXFederation(
            session_id=federation_session_id,
            computing_session=computing_session,
            party=party,
            parties=parties,
            max_message_size=max_message_size,
            mq=mq,
        )

    def __init__(
        self, session_id, computing_session, party: PartyMeta, parties: typing.List[PartyMeta], max_message_size, mq
    ):
        super().__init__(
            session_id=session_id,
            computing_session=computing_session,
            party=party,
            parties=parties,
            max_message_size=max_message_size,
            mq=mq,
        )

    def __getstate__(self):
        pass

    def destroy(self):
        LOGGER.debug("start to cleanup...")

        channel = MQChannel(
            host=self._mq.host,
            port=self._mq.port,
            namespace=self._session_id,
            send_topic=None,
            receive_topic=None,
            src_party_id=None,
            src_role=None,
            dst_party_id=None,
            dst_role=None,
        )

        channel.cleanup()
        channel.close()

    def _maybe_create_topic_and_replication(self, party, topic_suffix):
        LOGGER.debug(f"_maybe_create_topic_and_replication, party={party}, topic_suffix={topic_suffix}")
        send_topic_name = f"{self._session_id}-{self._party.role}-{self._party.party_id}-{party.role}-{party.party_id}-{topic_suffix}"
        receive_topic_name = f"{self._session_id}-{party.role}-{party.party_id}-{self._party.role}-{self._party.party_id}-{topic_suffix}"

        # topic_pair is a pair of topic for sending and receiving message respectively
        topic_pair = _TopicPair(
            namespace=self._session_id,
            send=send_topic_name,
            receive=receive_topic_name,
        )
        return topic_pair

    def _get_channel(
        self, topic_pair: _TopicPair, src_party_id, src_role, dst_party_id, dst_role, mq: MQ, conf: dict = None
    ):
        LOGGER.debug(
            f"_get_channel, topic_pari={topic_pair}, src_party_id={src_party_id}, src_role={src_role}, dst_party_id={dst_party_id}, dst_role={dst_role}"
        )
        return MQChannel(
            host=mq.host,
            port=mq.port,
            namespace=topic_pair.namespace,
            send_topic=topic_pair.send,
            receive_topic=topic_pair.receive,
            src_party_id=src_party_id,
            src_role=src_role,
            dst_party_id=dst_party_id,
            dst_role=dst_role,
        )

    _topic_ip_map = {}

    @nretry
    def _query_receive_topic(self, channel_info):
        LOGGER.debug(f"_query_receive_topic, channel_info={channel_info}")
        topic = channel_info._receive_topic
        if topic not in self._topic_ip_map:
            LOGGER.info(f"query topic {topic} miss cache ")
            response = channel_info.query()
            if response.code == "0":
                topic_info = osx_pb2.TopicInfo()
                topic_info.ParseFromString(response.payload)
                self._topic_ip_map[topic] = (topic_info.ip, topic_info.port)
                LOGGER.info(f"query result {topic} {topic_info}")
            else:
                raise LookupError(f"{response}")
        host, port = self._topic_ip_map[topic]

        new_channel_info = channel_info
        if channel_info._host != host or channel_info._port != port:
            LOGGER.info(
                f"channel info missmatch, host: {channel_info._host} vs {host} and port: {channel_info._port} vs {port}"
            )
            new_channel_info = MQChannel(
                host=host,
                port=port,
                namespace=channel_info._namespace,
                send_topic=channel_info._send_topic,
                receive_topic=channel_info._receive_topic,
                src_party_id=channel_info._src_party_id,
                src_role=channel_info._src_role,
                dst_party_id=channel_info._dst_party_id,
                dst_role=channel_info._dst_role,
            )
        return new_channel_info

    def _get_consume_message(self, channel_info):
        LOGGER.debug(f"_get_comsume_message, channel_info={channel_info}")
        while True:
            response = channel_info.consume()
            LOGGER.debug(f"_get_comsume_message, channel_info={channel_info}, response={response}")
            # if response.code == "138":
            #     continue
            message = osx_pb2.Message()
            message.ParseFromString(response.payload)
            offset = response.metadata["MessageOffSet"]
            head_str = str(message.head, encoding="utf-8")
            LOGGER.debug(f"head str {head_str}")
            properties = json.loads(head_str)
            LOGGER.debug(f"osx response properties {properties}")
            body = message.body
            yield offset, properties, body

    def _consume_ack(self, channel_info, id):
        LOGGER.debug(f"_comsume_ack, channel_info={channel_info}, id={id}")
        channel_info.ack(offset=id)
