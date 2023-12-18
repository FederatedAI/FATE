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

from fate.arch.federation.api import PartyMeta
from fate.arch.federation.backends.osx import osx_pb2
from fate.arch.federation.message_queue import MessageQueueBasedFederation
from ._mq_channel import MQChannel

LOGGER = getLogger(__name__)


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


class OSXFederation(MessageQueueBasedFederation):
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
        mq = MQ(host, port)

        return OSXFederation(
            federation_session_id=federation_session_id,
            computing_session=computing_session,
            party=party,
            parties=parties,
            max_message_size=max_message_size,
            mq=mq,
        )

    def __init__(
        self,
        federation_session_id,
        computing_session,
        party: PartyMeta,
        parties: typing.List[PartyMeta],
        max_message_size,
        mq,
    ):
        super().__init__(
            session_id=federation_session_id,
            computing_session=computing_session,
            party=party,
            parties=parties,
            max_message_size=max_message_size,
            mq=mq,
        )

    def __getstate__(self):
        pass

    def _destroy(self):
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

    def _get_channel(self, topic_pair, src_party_id, src_role, dst_party_id, dst_role, mq=None, conf: dict = None):
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

    def _get_consume_message(self, channel_info):
        LOGGER.debug(f"_get_comsume_message, channel_info={channel_info}")
        while True:
            response = channel_info.consume()
            # LOGGER.debug(f"_get_comsume_message, channel_info={channel_info}, response={response}")
            if response.code == "E0000000601":
                raise LookupError(f"{response}")
            message = osx_pb2.Message()
            message.ParseFromString(response.payload)

            head_str = str(message.head, encoding="utf-8")
            # LOGGER.debug(f"head str {head_str}")
            properties = json.loads(head_str)
            # LOGGER.debug(f"osx response properties {properties}")
            body = message.body
            yield 0, properties, body

    def _consume_ack(self, channel_info, id):
        return
