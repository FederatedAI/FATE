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
import grpc
import json
import uuid
import collections
from concurrent import futures
from grpc._cython import cygrpc
from fate_flow.utils.proto_compatibility import proxy_pb2_grpc
from fate_flow.utils.grpc_utils import wrap_grpc_packet
import time


class _ClientCallDetails(
    collections.namedtuple(
        '_ClientCallDetails',
        ('method', 'timeout', 'metadata', 'credentials', 'wait_for_ready')),
    grpc.ClientCallDetails):
    pass


class RequestIDClientInterceptor(grpc.UnaryUnaryClientInterceptor):

    def intercept_unary_unary(self, continuation, client_call_details, request):
        rid = uuid.uuid1()
        print(f"Sending RPC request, Method: {client_call_details.method}, Request ID: {rid}.")

        # Add request into client call details, aka, metadata.
        metadata = []
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)
        metadata.append(("request_id", rid))

        client_call_details = _ClientCallDetails(
            client_call_details.method, client_call_details.timeout, metadata,
            client_call_details.credentials, client_call_details.wait_for_ready)
        return continuation(client_call_details, request)


class UnaryServicer(proxy_pb2_grpc.DataTransferServiceServicer):
    def unaryCall(self, _request, context):
        packet = _request
        header = packet.header
        _suffix = packet.body.key
        param_bytes = packet.body.value
        param = bytes.decode(param_bytes)
        job_id = header.task.taskId
        src = header.src
        dst = header.dst
        method = header.operator
        resp_json = "xczczcz"

        for key, value in context.invocation_metadata():
            print(key, value)

        context.set_trailing_metadata((
            ('retcode', "0"),
            ('msg', 'ok'),
        ))

        return wrap_grpc_packet(resp_json, method, _suffix, dst.partyId, src.partyId, job_id)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                                  (cygrpc.ChannelArgKey.max_receive_message_length, -1)])

    proxy_pb2_grpc.add_DataTransferServiceServicer_to_server(UnaryServicer(), server)
    server.add_insecure_port("{}:{}".format("0.0.0.0", 9360))
    server.start()
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
