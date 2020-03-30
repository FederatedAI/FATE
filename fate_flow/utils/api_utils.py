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

import requests
from flask import jsonify
from flask import Response

from fate_flow.entity.constant_config import WorkMode
from fate_flow.settings import DEFAULT_GRPC_OVERALL_TIMEOUT, CHECK_NODES_IDENTITY, MANAGER_HOST, MANAGER_PORT, \
    FATE_MANAGER_GET_NODE_INFO, HEADERS, audit_logger
from fate_flow.utils.grpc_utils import wrap_grpc_packet, get_proxy_data_channel
from fate_flow.entity.runtime_config import RuntimeConfig


def get_json_result(retcode=0, retmsg='success', data=None, job_id=None, meta=None):
    result_dict = {"retcode": retcode, "retmsg": retmsg, "data": data, "jobId": job_id, "meta": meta}
    response = {}
    for key, value in result_dict.items():
        if not value and key != "retcode":
            continue
        else:
            response[key] = value
    return jsonify(response)


def error_response(response_code, retmsg):
    return Response(json.dumps({'retmsg': retmsg, 'retcode': response_code}), status=response_code, mimetype='application/json')


def federated_api(job_id, method, endpoint, src_party_id, dest_party_id, src_role, json_body, work_mode,
                  overall_timeout=DEFAULT_GRPC_OVERALL_TIMEOUT):
    if int(dest_party_id) == 0:
        return local_api(method=method, endpoint=endpoint, json_body=json_body)
    if work_mode == WorkMode.STANDALONE:
        return local_api(method=method, endpoint=endpoint, json_body=json_body)
    elif work_mode == WorkMode.CLUSTER:
        return remote_api(job_id=job_id, method=method, endpoint=endpoint, src_party_id=src_party_id, src_role=src_role,
                          dest_party_id=dest_party_id, json_body=json_body, overall_timeout=overall_timeout)
    else:
        raise Exception('{} work mode is not supported'.format(work_mode))


def remote_api(job_id, method, endpoint, src_party_id, dest_party_id, src_role, json_body,
               overall_timeout=DEFAULT_GRPC_OVERALL_TIMEOUT):
    json_body['src_role'] = src_role
    if CHECK_NODES_IDENTITY:
        get_node_identity(json_body, src_party_id)
    _packet = wrap_grpc_packet(json_body, method, endpoint, src_party_id, dest_party_id, job_id,
                               overall_timeout=overall_timeout)
    try:
        channel, stub = get_proxy_data_channel()
        _return = stub.unaryCall(_packet)
        audit_logger.info("grpc api response: {}".format(_return))
        channel.close()
        json_body = json.loads(_return.body.value)
        return json_body
    except Exception as e:
        raise Exception('rpc request error: {}'.format(e))


def local_api(method, endpoint, json_body):
    try:
        url = "http://{}{}".format(RuntimeConfig.JOB_SERVER_HOST, endpoint)
        audit_logger.info('local api request: {}'.format(url))
        action = getattr(requests, method.lower(), None)
        response = action(url=url, json=json_body, headers=HEADERS)
        audit_logger.info(response.text)
        response_json_body = response.json()
        audit_logger.info('local api response: {} {}'.format(endpoint, response_json_body))
        return response_json_body
    except Exception as e:
        raise Exception('local request error: {}'.format(e))


def request_execute_server(request, execute_host):
    try:
        endpoint = request.base_url.replace(request.host_url, '')
        method = request.method
        url = "http://{}/{}".format(execute_host, endpoint)
        audit_logger.info('sub request: {}'.format(url))
        action = getattr(requests, method.lower(), None)
        response = action(url=url, json=request.json, headers=HEADERS)
        return jsonify(response.json())
    except requests.exceptions.ConnectionError as e:
        return get_json_result(retcode=999, retmsg='please start execute server: {}'.format(execute_host))
    except Exception as e:
        raise Exception('local request error: {}'.format(e))


def get_node_identity(json_body, src_party_id):
    params = {
        'partyId': src_party_id
    }
    try:
        response = requests.get(url="http://{}:{}{}".format(MANAGER_HOST, MANAGER_PORT, FATE_MANAGER_GET_NODE_INFO), params=params)
        json_body['appKey'] = response.json().get('data').get('appKey')
        json_body['appSecret'] = response.json().get('data').get('appSecret')
    except Exception as e:
        raise Exception('get appkey and secret failed: {}'.format(str(e)))
