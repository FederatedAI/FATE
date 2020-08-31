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

import grpc

from fate_flow.settings import FATEFLOW_SERVICE_NAME, IP, GRPC_PORT, DEFAULT_GRPC_OVERALL_TIMEOUT
from fate_flow.utils.proto_compatibility import basic_meta_pb2
from fate_flow.utils.proto_compatibility import proxy_pb2, proxy_pb2_grpc


def get_proxy_data_channel():
    channel = grpc.insecure_channel('{}:{}'.format("127.0.0.1", 9361))
    stub = proxy_pb2_grpc.DataTransferServiceStub(channel)
    return channel, stub


def wrap_grpc_packet(_json_body, _method, _url, _src_party_id, _dst_party_id, job_id=None, overall_timeout=DEFAULT_GRPC_OVERALL_TIMEOUT):
    _src_end_point = basic_meta_pb2.Endpoint(ip=IP, port=GRPC_PORT)
    _src = proxy_pb2.Topic(name=job_id, partyId="{}".format(_src_party_id), role=FATEFLOW_SERVICE_NAME, callback=_src_end_point)
    _dst = proxy_pb2.Topic(name=job_id, partyId="{}".format(_dst_party_id), role=FATEFLOW_SERVICE_NAME, callback=None)
    _task = proxy_pb2.Task(taskId=job_id)
    _command = proxy_pb2.Command(name=FATEFLOW_SERVICE_NAME)
    _conf = proxy_pb2.Conf(overallTimeout=overall_timeout)
    _meta = proxy_pb2.Metadata(src=_src, dst=_dst, task=_task, command=_command, operator=_method, conf=_conf)
    _data = proxy_pb2.Data(key=_url, value=bytes(json.dumps(_json_body), 'utf-8'))
    return proxy_pb2.Packet(header=_meta, body=_data)


def send():
    json_body = "dadadad"
    method = "POST"
    endpoint = "test"
    src_party_id = 9999
    dest_party_id = 10000
    job_id = "121"
    _packet = wrap_grpc_packet(json_body, method, endpoint, src_party_id, dest_party_id, job_id)
    try:
        channel, stub = get_proxy_data_channel()
        _return, call = stub.unaryCall.with_call(_packet, metadata=(
            ('k1', 'v1'),
            ('k2', 'v2'),
        ))
        print("grpc api response: {}".format(_return))

        for key, value in call.trailing_metadata():
            print(key, value)
        """
        _return, call = stub.unaryCall.with_call(_packet)
        """
        channel.close()

        json_body = json.loads(_return.body.value)
        return json_body
    except Exception as e:
        raise Exception('rpc request error: {}'.format(e))


if __name__ == '__main__':
    send()
