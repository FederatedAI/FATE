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

import json

from fate_arch.common import Party
from fate_arch.common.log import getLogger
from fate_arch.federation._federation_base import FederationBase
from fate_arch.federation._nretry import nretry
from fate_arch.federation.firework._mq_channel import MQChannel
from fate_arch.protobuf.python import osx_pb2

LOGGER = getLogger()
# default message max size in bytes = 1MB
DEFAULT_MESSAGE_MAX_SIZE = 104857


class MQ(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def __str__(self):
        return (
            f"MQ(host={self.host}, port={self.port}, "
            f"type=firework"
        )

    def __repr__(self):
        return self.__str__()


class _TopicPair(object):
    def __init__(self, namespace, send, receive):
        self.namespace = namespace
        self.send = send
        self.receive = receive


class Federation(FederationBase):

    @staticmethod
    def from_conf(federation_session_id: str,
                  party: Party,
                  runtime_conf: dict,
                  **kwargs):
        firework_config = kwargs["firework_config"]
        LOGGER.debug(f"firework_config: {firework_config}")
        host = firework_config.get("host", "localhost")
        port = firework_config.get("port", "6650")

        firework_run = runtime_conf.get(
            "job_parameters", {}).get("firework_run", {})
        LOGGER.debug(f"firework_run: {firework_run}")

        max_message_size = firework_run.get(
            "max_message_size", DEFAULT_MESSAGE_MAX_SIZE)

        LOGGER.debug(f"set max message size to {max_message_size} Bytes")

        mq = MQ(host, port)

        return Federation(federation_session_id, party, mq)

    def __init__(self, session_id, party: Party, mq):
        super().__init__(session_id=session_id, party=party, mq=mq)


    def __getstate__(self):
        pass

    def cleanup(self, parties):
        LOGGER.debug("[firework.cleanup]start to cleanup...")

        channel = MQChannel(
            host=self._mq.host,
            port=self._mq.port,
            namespace=self._session_id,
            send_topic=None,
            receive_topic=None,
            src_party_id=None,
            src_role=None,
            dst_party_id=None,
            dst_role=None
        )

        channel.cleanup()
        channel.close()

    def _maybe_create_topic_and_replication(self, party, topic_suffix):
        send_topic_name = f"{self._session_id}-{self._party.role}-{self._party.party_id}-{party.role}-{party.party_id}-{topic_suffix}"
        receive_topic_name = f"{self._session_id}-{party.role}-{party.party_id}-{self._party.role}-{self._party.party_id}-{topic_suffix}"

        # topic_pair is a pair of topic for sending and receiving message respectively
        topic_pair = _TopicPair(
            namespace=self._session_id,
            send=send_topic_name,
            receive=receive_topic_name,
        )
        return topic_pair

    def _get_channel(self,
                     topic_pair: _TopicPair,
                     src_party_id,
                     src_role,
                     dst_party_id,
                     dst_role,
                     mq=None,
                     conf: dict = None):
        return MQChannel(
            host=mq.host,
            port=mq.port,
            namespace=topic_pair.namespace,
            send_topic=topic_pair.send,
            receive_topic=topic_pair.receive,
            src_party_id=src_party_id,
            src_role=src_role,
            dst_party_id=dst_party_id,
            dst_role=dst_role
        )

    _topic_ip_map = {}
    @nretry
    def _query_receive_topic(self, channel_info):

        new_channel_info = channel_info
        if self._topic_ip_map.__contains__(channel_info._receive_topic):
            LOGGER.info("query topic hit cache")
            host = self._topic_ip_map[channel_info._receive_topic][0]
            port = self._topic_ip_map[channel_info._receive_topic][1]
        else:
            response = channel_info.query()
            if response.code == 0:
                topic_info= osx_pb2.TopicInfo()
                topic_info.ParseFromString(response.payload)
                host = topic_info.ip
                port = topic_info.port
                self._topic_ip_map[channel_info._receive_topic] = (host,port)
                LOGGER.info(f"query result {channel_info._receive_topic} {host} : {port}")
            else:
                raise LookupError
        if channel_info._host != host or channel_info._port != port:
            new_channel_info = MQChannel(
                host=host,
                port=port,
                namespace=channel_info._namespace,
                send_topic=channel_info._send_topic,
                receive_topic=channel_info._receive_topic,
                src_party_id=channel_info._src_party_id,
                src_role=channel_info._src_role,
                dst_party_id=channel_info._dst_party_id,
                dst_role=channel_info._dst_role
            )
        return new_channel_info

    def _get_consume_message(self, channel_info):
        while True:
            response = channel_info.consume()
            message = osx_pb2.Message()
            message.ParseFromString(response.payload)
            offset = response.metadata["MessageOffSet"]
            head_str = str(message.head, encoding="utf-8")
            LOGGER.debug(f"head str {head_str}")
            properties = json.loads(head_str)
            LOGGER.info(f"osx response properties {properties}")
            body = message.body
            yield offset, properties, body

    def _consume_ack(self, channel_info, id):
        channel_info.ack(offset=id)

if __name__ == "__main__":
    mq = MQ("localhost", 9370)
    federation = Federation("", Party("testRole",9999), mq)

