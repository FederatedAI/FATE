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
import requests
import json
from fate_flow.utils.proto_compatibility import basic_meta_pb2
from fate_flow.utils.proto_compatibility import proxy_pb2, proxy_pb2_grpc
import grpc

from fate_flow.settings import ROLE, IP, GRPC_PORT, PROXY_HOST, PROXY_PORT, HEADERS, DEFAULT_GRPC_OVERALL_TIMEOUT, \
    audit_logger
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.utils.node_check_utils import nodes_check


def get_proxy_data_channel():
    channel = grpc.insecure_channel('{}:{}'.format(PROXY_HOST, PROXY_PORT))
    stub = proxy_pb2_grpc.DataTransferServiceStub(channel)
    return channel, stub


def wrap_grpc_packet(_json_body, _method, _url, _src_party_id, _dst_party_id, job_id=None, overall_timeout=DEFAULT_GRPC_OVERALL_TIMEOUT):
    _src_end_point = basic_meta_pb2.Endpoint(ip=IP, port=GRPC_PORT)
    _src = proxy_pb2.Topic(name=job_id, partyId="{}".format(_src_party_id), role=ROLE, callback=_src_end_point)
    _dst = proxy_pb2.Topic(name=job_id, partyId="{}".format(_dst_party_id), role=ROLE, callback=None)
    _task = proxy_pb2.Task(taskId=job_id)
    _command = proxy_pb2.Command(name=ROLE)
    _conf = proxy_pb2.Conf(overallTimeout=overall_timeout)
    _meta = proxy_pb2.Metadata(src=_src, dst=_dst, task=_task, command=_command, operator=_method, conf=_conf)
    _data = proxy_pb2.Data(key=_url, value=bytes(json.dumps(_json_body), 'utf-8'))
    return proxy_pb2.Packet(header=_meta, body=_data)


def get_url(_suffix):
    return "http://{}/{}".format(RuntimeConfig.JOB_SERVER_HOST.rstrip('/'), _suffix.lstrip('/'))


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
        param_dict = json.loads(param)
        param_dict['src_party_id'] = str(src.partyId)
        try:
            nodes_check(param_dict.get('src_party_id'), param_dict.get('src_role'), param_dict.get('appKey'),
                        param_dict.get('appSecret'))
        except Exception as e:
            resp_json = {
                "retcode": 100,
                "retmsg": str(e)
            }
            return wrap_grpc_packet(resp_json, method, _suffix, dst.partyId, src.partyId, job_id)
        param = bytes.decode(bytes(json.dumps(param_dict), 'utf-8'))

        action = getattr(requests, method.lower(), None)
        audit_logger.info('rpc receive: {}'.format(packet))
        if action:
            audit_logger.info("rpc receive: {} {}".format(get_url(_suffix), param))
            resp = action(url=get_url(_suffix), data=param, headers=HEADERS)
        else:
            pass
        resp_json = resp.json()
        return wrap_grpc_packet(resp_json, method, _suffix, dst.partyId, src.partyId, job_id)