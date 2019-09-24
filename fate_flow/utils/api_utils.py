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
import requests
from flask import jsonify
from flask import Response

from fate_flow.entity.constant_config import WorkMode
from fate_flow.settings import DEFAULT_GRPC_OVERALL_TIMEOUT
from fate_flow.settings import stat_logger, HEADERS
from fate_flow.utils.grpc_utils import wrap_grpc_packet, get_proxy_data_channel
from fate_flow.entity.runtime_config import RuntimeConfig


def get_json_result(retcode=0, retmsg='success', data=None, job_id=None, meta=None):
    return jsonify({"retcode": retcode, "retmsg": retmsg, "data": data, "jobId": job_id, "meta": meta})


def error_response(response_code, retmsg):
    return Response(json.dumps({'retmsg': retmsg, 'retcode': response_code}), status=response_code, mimetype='application/json')


def federated_api(job_id, method, endpoint, src_party_id, dest_party_id, json_body, work_mode,
                  overall_timeout=DEFAULT_GRPC_OVERALL_TIMEOUT):
    if dest_party_id == 0:
        return local_api(method=method, endpoint=endpoint, json_body=json_body)
    if work_mode == WorkMode.STANDALONE:
        return local_api(method=method, endpoint=endpoint, json_body=json_body)
    elif work_mode == WorkMode.CLUSTER:
        return remote_api(job_id=job_id, method=method, endpoint=endpoint, src_party_id=src_party_id,
                          dest_party_id=dest_party_id, json_body=json_body, overall_timeout=overall_timeout)
    else:
        raise Exception('{} work mode is not supported'.format(work_mode))


def remote_api(job_id, method, endpoint, src_party_id, dest_party_id, json_body,
               overall_timeout=DEFAULT_GRPC_OVERALL_TIMEOUT):
    _packet = wrap_grpc_packet(json_body, method, endpoint, src_party_id, dest_party_id, job_id,
                               overall_timeout=overall_timeout)
    try:
        channel, stub = get_proxy_data_channel()
        # stat_logger.info("grpc api request: {}".format(_packet))
        _return = stub.unaryCall(_packet)
        stat_logger.info("grpc api response: {}".format(_return))
        channel.close()
        json_body = json.loads(_return.body.value)
        return json_body
    except grpc.RpcError as e:
        raise Exception('rpc request error: {}'.format(e))
    except Exception as e:
        raise Exception('rpc request error: {}'.format(e))


def local_api(method, endpoint, json_body):
    try:
        url = "http://{}{}".format(RuntimeConfig.JOB_SERVER_HOST, endpoint)
        stat_logger.info('local api request: {}'.format(url))
        action = getattr(requests, method.lower(), None)
        response = action(url=url, json=json_body, headers=HEADERS)
        stat_logger.info(response.text)
        response_json_body = response.json()
        stat_logger.info('local api response: {} {}'.format(endpoint, response_json_body))
        return response_json_body
    except Exception as e:
        raise Exception('local request error: {}'.format(e))


def request_execute_server(request, execute_host):
    try:
        endpoint = request.base_url.replace(request.host_url, '')
        method = request.method
        url = "http://{}/{}".format(execute_host, endpoint)
        stat_logger.info('sub request: {}'.format(url))
        action = getattr(requests, method.lower(), None)
        response = action(url=url, json=request.json, headers=HEADERS)
        return jsonify(response.json())
    except requests.exceptions.ConnectionError as e:
        return get_json_result(retcode=999, retmsg='please start execute server: {}'.format(execute_host))
    except Exception as e:
        raise Exception('local request error: {}'.format(e))
