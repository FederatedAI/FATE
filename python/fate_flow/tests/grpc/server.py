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
import requests
from grpc._cython import cygrpc

from fate_arch.common.base_utils import json_dumps, json_loads
from fate_flow.db.runtime_config import RuntimeConfig
from fate_flow.settings import FATE_FLOW_SERVICE_NAME
from fate_flow.settings import stat_logger, HOST, GRPC_PORT
from fate_flow.db.job_default_config import JobDefaultConfig
from fate_flow.utils.proto_compatibility import basic_meta_pb2
from fate_flow.utils.proto_compatibility import proxy_pb2
from fate_flow.utils.proto_compatibility import proxy_pb2_grpc
import time
import sys
from fate_flow.tests.grpc.xthread import ThreadPoolExecutor


def wrap_grpc_packet(json_body, http_method, url, src_party_id, dst_party_id, job_id=None, overall_timeout=None):
    overall_timeout = JobDefaultConfig.remote_request_timeout if overall_timeout is None else overall_timeout
    _src_end_point = basic_meta_pb2.Endpoint(ip=HOST, port=GRPC_PORT)
    _src = proxy_pb2.Topic(name=job_id, partyId="{}".format(src_party_id), role=FATE_FLOW_SERVICE_NAME, callback=_src_end_point)
    _dst = proxy_pb2.Topic(name=job_id, partyId="{}".format(dst_party_id), role=FATE_FLOW_SERVICE_NAME, callback=None)
    _task = proxy_pb2.Task(taskId=job_id)
    _command = proxy_pb2.Command(name=FATE_FLOW_SERVICE_NAME)
    _conf = proxy_pb2.Conf(overallTimeout=overall_timeout)
    _meta = proxy_pb2.Metadata(src=_src, dst=_dst, task=_task, command=_command, operator=http_method, conf=_conf)
    _data = proxy_pb2.Data(key=url, value=bytes(json_dumps(json_body), 'utf-8'))
    return proxy_pb2.Packet(header=_meta, body=_data)


def get_url(_suffix):
    return "http://{}:{}/{}".format(RuntimeConfig.JOB_SERVER_HOST, RuntimeConfig.HTTP_PORT, _suffix.lstrip('/'))


class UnaryService(proxy_pb2_grpc.DataTransferServiceServicer):
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
        param_dict = json_loads(param)
        param_dict['src_party_id'] = str(src.partyId)
        source_routing_header = []
        for key, value in context.invocation_metadata():
            source_routing_header.append((key, value))
        stat_logger.info(f"grpc request routing header: {source_routing_header}")

        param = bytes.decode(bytes(json_dumps(param_dict), 'utf-8'))

        action = getattr(requests, method.lower(), None)
        if action:
            print(_suffix)
            #resp = action(url=get_url(_suffix), data=param, headers=HEADERS)
        else:
            pass
        #resp_json = resp.json()
        resp_json = {"status": "test"}
        import time
        print("sleep")
        time.sleep(60)
        return wrap_grpc_packet(resp_json, method, _suffix, dst.partyId, src.partyId, job_id)

thread_pool_executor = ThreadPoolExecutor(max_workers=5)
print(f"start grpc server pool on {thread_pool_executor._max_workers} max workers")
server = grpc.server(thread_pool_executor,
                     options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                              (cygrpc.ChannelArgKey.max_receive_message_length, -1)])

proxy_pb2_grpc.add_DataTransferServiceServicer_to_server(UnaryService(), server)
server.add_insecure_port("{}:{}".format("127.0.0.1", 9360))
server.start()

try:
    while True:
        time.sleep(60 * 60 * 24)
except KeyboardInterrupt:
    server.stop(0)
    sys.exit(0)
