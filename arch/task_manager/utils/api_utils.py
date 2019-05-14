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
from arch.task_manager.settings import logger, LOCAL_URL, HEADERS
from flask import jsonify
import json
from arch.task_manager.utils.grpc_utils import wrap_grpc_packet, get_proxy_data_channel
from arch.task_manager.job_manager import generate_job_id
from arch.task_manager.settings import PARTY_ID, DEFAULT_GRPC_OVERALL_TIMEOUT
import requests
import grpc


def get_json_result(status=0, msg='success', data=None, job_id=None):
    return jsonify({"status": status, "msg": msg, "data": data, "jobId": job_id})


def federated_api(job_id, method, url, party_id, json_body={}, overall_timeout=DEFAULT_GRPC_OVERALL_TIMEOUT):
    _packet = wrap_grpc_packet(json_body, method, url, party_id, job_id, overall_timeout=overall_timeout)
    try:
        channel, stub = get_proxy_data_channel()
        _return = stub.unaryCall(_packet)
        logger.info("grpc unary response: {}".format(_return))
        channel.close()
        return 0, _return.body.value
    except grpc.RpcError as e:
        logger.exception(e)
        return 101, 'rpc error'
    except Exception as e:
        logger.exception(e)
        return 102, str(e)


def new_federated_job(request, overall_timeout=DEFAULT_GRPC_OVERALL_TIMEOUT):
    request_config = request.json
    _job_id = generate_job_id()
    st, msg = federated_api(job_id=_job_id,
                            method='POST',
                            url='/{}/do'.format(request.base_url.replace(request.host_url, '')),
                            party_id=request_config.get('local', {}).get('party_id', PARTY_ID),
                            json_body=request_config,
                            overall_timeout=overall_timeout
                            )
    if st == 0:
        json_body = json.loads(msg)
        return get_json_result(status=json_body['status'], msg=json_body['msg'], data=json_body.get('data'), job_id=json_body['jobId'])
    else:
        return get_json_result(status=st, msg=msg, job_id=_job_id)


def local_api(method, suffix, data=None, json_body=None):
    url = "{}{}".format(LOCAL_URL, suffix)
    action = getattr(requests, method.lower(), None)
    resp = action(url=url, data=data, json=json_body, headers=HEADERS)
    return resp.json()
