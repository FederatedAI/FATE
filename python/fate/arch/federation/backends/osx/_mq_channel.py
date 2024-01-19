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
import time
from enum import Enum
from logging import getLogger
from typing import Dict, List, Any

import grpc
import numpy as np

from fate.arch.federation.backends.osx import osx_pb2
from fate.arch.federation.backends.osx.osx_pb2_grpc import PrivateTransferTransportStub

LOGGER = getLogger(__name__)


class Metadata(Enum):
    PTP_VERSION = "x-ptp-version"
    PTP_TECH_PROVIDER_CODE = "x-ptp-tech-provider-code"
    PTP_TRACE_ID = "x-ptp-trace-id"
    PTP_TOKEN = "x-ptp-token"
    PTP_URI = "x-ptp-uri"
    PTP_FROM_NODE_ID = "x-ptp-from-node-id"
    PTP_FROM_INST_ID = "x-ptp-from-inst-id"
    PTP_TARGET_NODE_ID = "x-ptp-target-node-id"
    PTP_TARGET_INST_ID = "x-ptp-target-inst-id"
    PTP_SESSION_ID = "x-ptp-session-id"
    PTP_TOPIC = "x-ptp-topic"
    PTP_TIMEOUT = "x-ptp-timeout"

    def key(self) -> str:
        return self.value

    def set(self, attachments: Dict[str, str], v: str):
        if attachments and "" != v and v != attachments.get(self.key(), ""):
            attachments[self.key()] = v

    def get(self, attachments: Dict[str, str]) -> str:
        return attachments.get(self.key(), "")

    def append(self, attachments: List[Any], v: str):
        if attachments is not None and "" != v:
            attachments.append((self.key(), v))


def build_trace_id():
    timestamp = int(time.time())
    timestamp_str = str(timestamp)
    return timestamp_str + "_" + str(np.random.randint(10000))


class MQChannel(object):
    def __init__(
        self, host, port, namespace, send_topic, receive_topic, src_party_id, src_role, dst_party_id, dst_role
    ):
        self._host = host
        self._port = port
        self._namespace = namespace
        self._send_topic = send_topic
        self._receive_topic = receive_topic
        # self._index = 1
        self._src_party_id = src_party_id
        self._src_role = src_role
        self._dst_party_id = dst_party_id
        self._dst_role = dst_role
        self._channel = None
        self._stub = None
        self._timeout = None

        if self._timeout is None:
            from fate.arch.config import cfg
            self._timeout = cfg.federation.osx.timeout

        LOGGER.debug(f"init, mq={self}")

    def __str__(self):
        return f"<MQChannel namespace={self._namespace}, host={self._host},port={self._port}, src=({self._src_role}, {self._src_party_id}), dst=({self._dst_role}, {self._dst_party_id}), send_topic={self._send_topic}, receive_topic={self._receive_topic}>"

    def prepare_metadata_consume(self):
        metadata = []
        # Metadata.PTP_TRACE_ID.append(metadata, )
        if not self._namespace is None:
            Metadata.PTP_SESSION_ID.append(metadata, self._namespace)
        # if not self._dst_party_id is None:
        #     Metadata.PTP_TARGET_NODE_ID.append(metadata, str(self._dst_party_id))
        if not self._src_party_id is None:
            Metadata.PTP_FROM_NODE_ID.append(metadata, str(self._src_party_id))
        # Metadata.PTP_TOPIC.append(metadata,str(self._receive_topic))
        Metadata.PTP_TECH_PROVIDER_CODE.append(metadata, "FATE")
        Metadata.PTP_TRACE_ID.append(metadata, build_trace_id())
        return metadata

    def prepare_metadata(
        self,
    ):
        metadata = []
        Metadata.PTP_TRACE_ID.append(metadata, build_trace_id())
        if not self._namespace is None:
            Metadata.PTP_SESSION_ID.append(metadata, self._namespace)
        if not self._dst_party_id is None:
            Metadata.PTP_TARGET_NODE_ID.append(metadata, str(self._dst_party_id))
        if not self._src_party_id is None:
            Metadata.PTP_FROM_NODE_ID.append(metadata, str(self._src_party_id))
        # Metadata.PTP_TOPIC.append(metadata,str(self._receive_topic))
        Metadata.PTP_TECH_PROVIDER_CODE.append(metadata, "FATE")
        return metadata

    # @nretry
    def consume(self):
        self._get_or_create_channel()
        inbound = osx_pb2.PopInbound(topic=self._receive_topic, timeout=self._timeout)
        metadata = self.prepare_metadata_consume()
        result = self._stub.pop(request=inbound, metadata=metadata)
        # LOGGER.debug(f"consume, result={result.code}, mq={self}")
        return result

    # @nretry
    def produce(self, body, properties):
        # LOGGER.debug(f"produce body={body}, properties={properties}, mq={self}")
        self._get_or_create_channel()
        msg = osx_pb2.Message(head=bytes(json.dumps(properties), encoding="utf-8"), body=body)
        inbound = osx_pb2.PushInbound(topic=self._send_topic, payload=msg.SerializeToString())
        metadata = self.prepare_metadata()

        result = self._stub.push(inbound, metadata=metadata)

        return result

    # @nretry
    def ack(self, offset):
        return

    def cleanup(self):
        LOGGER.debug(f"cancel channel")
        self._get_or_create_channel()
        inbound = osx_pb2.ReleaseInbound()
        metadata = self.prepare_metadata()

        result = self._stub.release(inbound, metadata=metadata)

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

        self._stub = PrivateTransferTransportStub(self._channel)
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


if __name__ == "__main__":
    timestamp = int(time.time())
    timestamp_str = str(timestamp)

    print(build_trace_id())
