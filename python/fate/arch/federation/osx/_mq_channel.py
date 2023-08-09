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
from logging import getLogger

import grpc
from fate.arch.federation.osx import osx_pb2
from fate.arch.federation.osx.osx_pb2_grpc import PrivateTransferProtocolStub

from .._nretry import nretry

LOGGER = getLogger(__name__)


class MQChannel(object):
    def __init__(
        self, host, port, namespace, send_topic, receive_topic, src_party_id, src_role, dst_party_id, dst_role
    ):
        self._host = host
        self._port = port
        self._namespace = namespace
        self._send_topic = send_topic
        self._receive_topic = receive_topic
        self._index = 1
        self._src_party_id = src_party_id
        self._src_role = src_role
        self._dst_party_id = dst_party_id
        self._dst_role = dst_role
        self._channel = None
        self._stub = None
        LOGGER.debug(f"init, mq={self}")

    def __str__(self):
        return f"<MQChannel namespace={self._namespace}, host={self._host},port={self._port}, src=({self._src_role}, {self._src_party_id}), dst=({self._dst_role}, {self._dst_party_id}), send_topic={self._send_topic}, receive_topic={self._receive_topic}>"

    @nretry
    def consume(self, offset=-1):
        LOGGER.debug(f"consume, offset={offset}, mq={self}")
        self._get_or_create_channel()
        meta = dict(
            MessageTopic=self._receive_topic,
            TechProviderCode="FATE",
            SourceNodeID=self._src_party_id,
            TargetNodeID=self._dst_party_id,
            TargetComponentName=self._dst_role,
            SourceComponentName=self._src_role,
            TargetMethod="CONSUME_MSG",
            SessionID=self._namespace,
            MessageOffSet=str(offset),
        )
        inbound = osx_pb2.Inbound(metadata=meta)
        LOGGER.debug(f"consume, inbound={inbound}, mq={self}")
        result = self._stub.invoke(inbound)
        LOGGER.debug(f"consume, result={result.code}, mq={self}")

        return result

    @nretry
    def query(self):
        LOGGER.debug(f"query, mq={self}")
        self._get_or_create_channel()
        meta = dict(
            MessageTopic=self._receive_topic,
            TechProviderCode="FATE",
            SourceNodeID=self._src_party_id,
            TargetNodeID=self._dst_party_id,
            TargetComponentName=self._dst_role,
            SourceComponentName=self._src_role,
            TargetMethod="QUERY_TOPIC",
            SessionID=self._namespace,
        )
        inbound = osx_pb2.Inbound(metadata=meta)
        LOGGER.debug(f"query, inbound={inbound}, mq={self}")
        result = self._stub.invoke(inbound)
        LOGGER.debug(f"query, result={result}, mq={self}")
        return result

    @nretry
    def produce(self, body, properties):
        # LOGGER.debug(f"produce body={body}, properties={properties}, mq={self}")
        self._get_or_create_channel()
        meta = dict(
            MessageTopic=self._send_topic,
            TechProviderCode="FATE",
            SourceNodeID=self._src_party_id,
            TargetNodeID=self._dst_party_id,
            TargetComponentName=self._dst_role,
            SourceComponentName=self._src_role,
            TargetMethod="PRODUCE_MSG",
            SessionID=self._namespace,
        )
        msg = osx_pb2.Message(head=bytes(json.dumps(properties), encoding="utf-8"), body=body)
        inbound = osx_pb2.Inbound(metadata=meta, payload=msg.SerializeToString())
        # LOGGER.debug(f"produce inbound={inbound}, mq={self}")
        result = self._stub.invoke(inbound)

        LOGGER.debug(f"produce {self._receive_topic}  index {self._index} result={result.code}, mq={self}")
        if result.code!="0":
            raise RuntimeError(f"produce msg error ,code : {result.code} msg : {result.message}")
        self._index+=1
        return result

    @nretry
    def ack(self, offset):
        LOGGER.debug(f"ack offset={offset}, mq={self}")
        self._get_or_create_channel()
        meta = dict(
            MessageTopic=self._receive_topic,
            TechProviderCode="FATE",
            SourceNodeID=self._src_party_id,
            TargetNodeID=self._dst_party_id,
            TargetComponentName=self._dst_role,
            SourceComponentName=self._src_role,
            TargetMethod="ACK_MSG",
            SessionID=self._namespace,
            MessageOffSet=offset,
        )
        inbound = osx_pb2.Inbound(metadata=meta)
        # LOGGER.debug(f"ack inbound={inbound}, mq={self}")
        result = self._stub.invoke(inbound)
        LOGGER.debug(f"ack result={result}, mq={self}")
        return result

    def cleanup(self):
        LOGGER.debug(f"cancel channel")

    def cancel(self):
        LOGGER.debug(f"cancel channel")

    def close(self):
        LOGGER.debug(f"close channel")

    def _get_or_create_channel(self):
        LOGGER.debug(f"call _get_or_create_channel, mq={self}")
        target = f"{self._host}:{self._port}"
        if self._check_alive():
            LOGGER.debug(f"channel alive, return, mq={self}")
            return

        LOGGER.debug(f"channel not alive, creating, mq={self}")
        self._channel = grpc.insecure_channel(
            target=target,
            options=[
                ("grpc.max_send_message_length", int((2 << 30) - 1)),
                ("grpc.max_receive_message_length", int((2 << 30) - 1)),
                ("grpc.max_metadata_size", 128 << 20),
                ("grpc.keepalive_time_ms", 7200 * 1000),
                ("grpc.keepalive_timeout_ms", 3600 * 1000),
                ("grpc.keepalive_permit_without_calls", int(False)),
                ("grpc.per_rpc_retry_buffer_size", int(16 << 20)),
                ("grpc.enable_retries", 1),
                (
                    "grpc.service_config",
                    '{ "retryPolicy":{ '
                    '"maxAttempts": 4, "initialBackoff": "0.1s", '
                    '"maxBackoff": "1s", "backoffMutiplier": 2, '
                    '"retryableStatusCodes": [ "UNAVAILABLE" ] } }',
                ),
            ],
        )

        self._stub = PrivateTransferProtocolStub(self._channel)
        LOGGER.debug(f"channel created, mq={self}")

    def _check_alive(self):
        status = (
            grpc._common.CYGRPC_CONNECTIVITY_STATE_TO_CHANNEL_CONNECTIVITY[
                self._channel._channel.check_connectivity_state(True)
            ]
            if self._channel is not None
            else None
        )
        LOGGER.debug(f"_check_alive: status={status}, mq={self}")

        if status == grpc.ChannelConnectivity.SHUTDOWN:
            LOGGER.debug(f"_check_alive: return True, mq={self}")
            return True
        else:
            LOGGER.debug(f"_check_alive: return False, mq={self}")
            return False
