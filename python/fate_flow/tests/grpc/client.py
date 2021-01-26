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
import sys

import time

from fate_arch.common.log import audit_logger, schedule_logger
from fate_flow.utils.grpc_utils import wrap_grpc_packet, gen_routing_metadata
from fate_flow.utils.proto_compatibility import proxy_pb2_grpc
import grpc


def get_command_federation_channel(host, port):
    print(f"connect {host}:{port}")
    channel = grpc.insecure_channel('{}:{}'.format(host, port))
    stub = proxy_pb2_grpc.DataTransferServiceStub(channel)
    return channel, stub


def remote_api(host, port, job_id, method, endpoint, src_party_id, dest_party_id, src_role, json_body, api_version="v1",
               overall_timeout=30*1000, try_times=3):
    endpoint = f"/{api_version}{endpoint}"
    json_body['src_role'] = src_role
    json_body['src_party_id'] = src_party_id
    _packet = wrap_grpc_packet(json_body, method, endpoint, src_party_id, dest_party_id, job_id,
                               overall_timeout=overall_timeout)
    print(_packet)
    _routing_metadata = gen_routing_metadata(src_party_id=src_party_id, dest_party_id=dest_party_id)
    exception = None
    for t in range(try_times):
        try:
            channel, stub = get_command_federation_channel(host, port)
            _return, _call = stub.unaryCall.with_call(_packet, metadata=_routing_metadata, timeout=(overall_timeout/1000))
            audit_logger(job_id).info("grpc api response: {}".format(_return))
            channel.close()
            response = json.loads(_return.body.value)
            return response
        except Exception as e:
            exception = e
            schedule_logger(job_id).warning(f"remote request {endpoint} error, sleep and try again")
            time.sleep(2 * (t+1))
    else:
        tips = 'Please check rollSite and fateflow network connectivity'
        raise Exception('{}rpc request error: {}'.format(tips, exception))

host = sys.argv[1]
port = int(sys.argv[2])
src_role = sys.argv[3]
src_party_id = sys.argv[4]
dest_party_id = sys.argv[5]
response = remote_api(host, port, "test_job_command", "POST", "/version/get", src_party_id, dest_party_id, src_role, {"src_role": src_role, "src_party_id": src_party_id})
print(response)

