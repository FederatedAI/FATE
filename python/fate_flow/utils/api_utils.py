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

from fate_arch.common.log import audit_logger
from fate_arch.common import FederatedMode
from fate_arch.common import conf_utils
from fate_flow.settings import DEFAULT_GRPC_OVERALL_TIMEOUT, CHECK_NODES_IDENTITY,\
    FATE_MANAGER_GET_NODE_INFO_ENDPOINT, HEADERS, API_VERSION
from fate_flow.utils.grpc_utils import wrap_grpc_packet, get_command_federation_channel, get_routing_metadata
from fate_flow.utils.service_utils import ServiceUtils
from fate_flow.entity.runtime_config import RuntimeConfig


def get_json_result(retcode=0, retmsg='success', data=None, job_id=None, meta=None):
    result_dict = {"retcode": retcode, "retmsg": retmsg, "data": data, "jobId": job_id, "meta": meta}
    response = {}
    for key, value in result_dict.items():
        if value is None and key != "retcode":
            continue
        else:
            response[key] = value
    return jsonify(response)


def error_response(response_code, retmsg):
    return Response(json.dumps({'retmsg': retmsg, 'retcode': response_code}), status=response_code, mimetype='application/json')


def federated_api(job_id, method, endpoint, src_party_id, dest_party_id, src_role, json_body, federated_mode, api_version=API_VERSION,
                  overall_timeout=DEFAULT_GRPC_OVERALL_TIMEOUT):
    if int(dest_party_id) == 0:
        federated_mode = FederatedMode.SINGLE
    if federated_mode == FederatedMode.SINGLE:
        return local_api(job_id=job_id, method=method, endpoint=endpoint, json_body=json_body, api_version=api_version)
    elif federated_mode == FederatedMode.MULTIPLE:
        return remote_api(job_id=job_id, method=method, endpoint=endpoint, src_party_id=src_party_id, src_role=src_role,
                          dest_party_id=dest_party_id, json_body=json_body, api_version=api_version, overall_timeout=overall_timeout)
    else:
        raise Exception('{} work mode is not supported'.format(federated_mode))


def remote_api(job_id, method, endpoint, src_party_id, dest_party_id, src_role, json_body, api_version=API_VERSION,
               overall_timeout=DEFAULT_GRPC_OVERALL_TIMEOUT, try_times=3):
    endpoint = f"/{api_version}{endpoint}"
    json_body['src_role'] = src_role
    if CHECK_NODES_IDENTITY:
        get_node_identity(json_body, src_party_id)
    _packet = wrap_grpc_packet(json_body, method, endpoint, src_party_id, dest_party_id, job_id,
                               overall_timeout=overall_timeout)
    _routing_metadata = get_routing_metadata(src_party_id=src_party_id, dest_party_id=dest_party_id)
    exception = None
    for t in range(try_times):
        try:
            engine, channel, stub = get_command_federation_channel()
            # _return = stub.unaryCall(_packet)
            _return, _call = stub.unaryCall.with_call(_packet, metadata=_routing_metadata)
            audit_logger(job_id).info("grpc api response: {}".format(_return))
            channel.close()
            response = json.loads(_return.body.value)
            return response
        except Exception as e:
            exception = e
    else:
        tips = ''
        if 'Error received from peer' in str(exception):
            tips = 'Please check if the fate flow server of the other party is started. '
        if 'failed to connect to all addresses' in str(exception):
            tips = 'Please check whether the rollsite service(port: 9370) is started. '
        raise Exception('{}rpc request error: {}'.format(tips, exception))


def local_api(job_id, method, endpoint, json_body, api_version=API_VERSION, try_times=3):
    endpoint = f"/{api_version}{endpoint}"
    exception = None
    for t in range(try_times):
        try:
            url = "http://{}:{}{}".format(RuntimeConfig.JOB_SERVER_HOST, RuntimeConfig.HTTP_PORT, endpoint)
            audit_logger(job_id).info('local api request: {}'.format(url))
            action = getattr(requests, method.lower(), None)
            http_response = action(url=url, json=json_body, headers=HEADERS)
            audit_logger(job_id).info(http_response.text)
            response = http_response.json()
            audit_logger(job_id).info('local api response: {} {}'.format(endpoint, response))
            return response
        except Exception as e:
            exception = e
    else:
        raise Exception('local request error: {}'.format(exception))


def get_node_identity(json_body, src_party_id):
    params = {
        'partyId': int(src_party_id),
        'federatedId': conf_utils.get_base_config("fatemanager", {}).get("federatedId")
    }
    try:
        response = requests.post(url="http://{}:{}{}".format(
            ServiceUtils.get_item("fatemanager", "host"),
            ServiceUtils.get_item("fatemanager", "port"),
            FATE_MANAGER_GET_NODE_INFO_ENDPOINT), json=params)
        json_body['appKey'] = response.json().get('data').get('appKey')
        json_body['appSecret'] = response.json().get('data').get('appSecret')
        json_body['_src_role'] = response.json().get('data').get('role')
    except Exception as e:
        raise Exception('get appkey and secret failed: {}'.format(str(e)))
