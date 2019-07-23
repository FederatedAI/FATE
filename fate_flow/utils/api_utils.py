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
from fate_flow.settings import stat_logger, LOCAL_URL, HEADERS
from flask import jsonify
import json
from fate_flow.utils.grpc_utils import wrap_grpc_packet, get_proxy_data_channel
from fate_flow.utils.job_utils import generate_job_id
from fate_flow.settings import DEFAULT_GRPC_OVERALL_TIMEOUT
import requests
import grpc


def get_json_result(retcode=0, retmsg='success', data=None, job_id=None, meta=None):
    return jsonify({"retcode": retcode, "retmsg": retmsg, "data": data, "jobId": job_id, "meta": meta})


def federated_api(job_id, method, url, src_party_id, dest_party_id, json_body={}, overall_timeout=DEFAULT_GRPC_OVERALL_TIMEOUT):
    _packet = wrap_grpc_packet(json_body, method, url, src_party_id, dest_party_id, job_id, overall_timeout=overall_timeout)
    try:
        channel, stub = get_proxy_data_channel()
        _return = stub.unaryCall(_packet)
        stat_logger.info("grpc unary response: {}".format(_return))
        channel.close()
        json_body = json.loads(_return.body.value)
        return json_body.get('retcode', 103), _return.body.value
    except grpc.RpcError as e:
        stat_logger.exception(e)
        return 101, 'rpc error'
    except Exception as e:
        stat_logger.exception(e)
        return 102, str(e)


def local_api(method, suffix, data=None, json_body=None):
    url = "{}{}".format(LOCAL_URL, suffix)
    action = getattr(requests, method.lower(), None)
    resp = action(url=url, data=data, json=json_body, headers=HEADERS)
    return resp.json()
