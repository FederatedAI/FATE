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
from logging import getLogger

import grpc
from fate.arch.federation.osx import osx_pb2, pcp_pb2
from fate.arch.federation.osx.pcp_pb2_grpc import PrivateTransferProtocolStub

from .._nretry import nretry

LOGGER = getLogger()


class MQChannel(object):
    def __init__(
        self, host, port, namespace, send_topic, receive_topic, src_party_id, src_role, dst_party_id, dst_role
    ):
        self._host = host
        self._port = port
        self._namespace = namespace
        self._send_topic = send_topic
        self._receive_topic = receive_topic
        self._src_party_id = src_party_id
        self._src_role = src_role
        self._dst_party_id = dst_party_id
        self._dst_role = dst_role
        self._channel = None
        self._stub = None

    @nretry
    def consume(self, offset=-1):
        self._get_or_create_channel()
        meta = dict(
            MessageTopic=self._send_topic,
            TechProviderCode="FT",
            SourceNodeID=self._src_party_id,
            TargetNodeID=self._dst_party_id,
            TargetComponentName=self._dst_role,
            SourceComponentName=self._src_role,
            TargetMethod="CONSUME_MSG",
            SessionID=self._namespace,
            MessageOffSet=str(offset),
        )
        inbound = pcp_pb2.Inbound(metadata=meta)
        result = self._stub.invoke(inbound)

        # print(result.metadata.MessageOffSet)
        # print(result.code)
        # print(result.metadata)

        print(result)
        print(result.code)
        return result

    # @nretry
    # def cleanup(self):
    #     self._get_or_create_channel()
    #     response = self._stub.cancelTransfer(
    #         firework_transfer_pb2.CancelTransferRequest(transferId=self._receive_topic, sessionId=self._namespace))
    #     return response
    #
    @nretry
    def query(self):
        self._get_or_create_channel()
        LOGGER.debug(f"try to query {self._receive_topic} session {self._namespace}")
        meta = dict(
            MessageTopic=self._receive_topic,
            TechProviderCode="FT",
            SourceNodeID=self._src_party_id,
            TargetNodeID=self._dst_party_id,
            TargetComponentName=self._dst_role,
            SourceComponentName=self._src_role,
            TargetMethod="QUERY_TOPIC",
            SessionID=self._namespace,
        )
        inbound = pcp_pb2.Inbound(metadata=meta)
        result = self._stub.invoke(inbound)

        print(result)
        return result

    @nretry
    def produce(self, body, properties):
        self._get_or_create_channel()
        meta = dict(
            MessageTopic=self._receive_topic,
            TechProviderCode="FT",
            SourceNodeID=self._src_party_id,
            TargetNodeID=self._dst_party_id,
            TargetComponentName=self._dst_role,
            SourceComponentName=self._src_role,
            TargetMethod="PRODUCE_MSG",
            SessionID=self._namespace,
        )
        msg = osx_pb2.Message(head=bytes(json.dumps(properties), encoding="utf-8"), body=body)
        inbound = pcp_pb2.Inbound(metadata=meta, payload=msg.SerializeToString())
        result = self._stub.invoke(inbound)
        print(result)
        return result

    @nretry
    def ack(self, offset):
        self._get_or_create_channel()
        meta = dict(
            MessageTopic=self._send_topic,
            TechProviderCode="FT",
            SourceNodeID=self._src_party_id,
            TargetNodeID=self._dst_party_id,
            TargetComponentName=self._dst_role,
            SourceComponentName=self._src_role,
            TargetMethod="ACK_MSG",
            SessionID=self._namespace,
            MessageOffSet=offset,
        )
        inbound = pcp_pb2.Inbound(metadata=meta)
        result = self._stub.invoke(inbound)
        # print(result)
        return result

    # def close(self):
    #     try:
    #         if self._channel:
    #             self._channel.close()
    #         self._channel = None
    #         self._stub = None
    #     except Exception as e:
    #         LOGGER.exception(e)
    #         self._stub = None
    #         self._channel = None
    #
    # def cancel(self):
    #     self.close()

    def _get_or_create_channel(self):
        target = "{}:{}".format(self._host, self._port)
        if self._check_alive():
            return

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

    def _check_alive(self):
        status = (
            grpc._common.CYGRPC_CONNECTIVITY_STATE_TO_CHANNEL_CONNECTIVITY[
                self._channel._channel.check_connectivity_state(True)
            ]
            if self._channel is not None
            else None
        )

        if status == grpc.ChannelConnectivity.SHUTDOWN:
            return True
        else:
            return False


if __name__ == "__main__":
    mq = MQChannel(
        host="localhost",
        port="9370",
        namespace="test",
        send_topic="testTopic",
        receive_topic="testReceiveTopic",
        src_party_id="9999",
        src_role="",
        dst_party_id="9999",
        dst_role="",
    )
    properties = dict(ccc="jjj")
    # mq.produce(body=bytes("kaideng===",encoding="utf-8"),properties=properties)

    # mq.consume();
    mq.query()
