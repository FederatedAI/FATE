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
import time
from flask import jsonify, Response
from werkzeug.http import HTTP_STATUS_CODES

from fate_arch.common.base_utils import json_loads, json_dumps
from fate_arch.common.log import audit_logger, schedule_logger
from fate_arch.common import FederatedMode, CoordinationProxyService, CoordinationCommunicationProtocol
from fate_flow.settings import CHECK_NODES_IDENTITY,\
    FATE_MANAGER_GET_NODE_INFO_ENDPOINT, HEADERS, API_VERSION, stat_logger, HOST, HTTP_PORT, PROXY, PROXY_PROTOCOL
from fate_flow.db.job_default_config import JobDefaultConfig
from fate_flow.db.service_registry import ServiceRegistry
from fate_flow.utils.grpc_utils import wrap_grpc_packet, get_command_federation_channel, gen_routing_metadata, \
    forward_grpc_packet
from fate_flow.db.runtime_config import RuntimeConfig
from fate_flow.entity.types import RetCode


def get_json_result(retcode=RetCode.SUCCESS, retmsg='success', data=None, job_id=None, meta=None):
    result_dict = {"retcode": retcode, "retmsg": retmsg, "data": data, "jobId": job_id, "meta": meta}
    response = {}
    for key, value in result_dict.items():
        if value is None and key != "retcode":
            continue
        else:
            response[key] = value
    return jsonify(response)


def server_error_response(e):
    stat_logger.exception(e)
    if len(e.args) > 1:
        return get_json_result(retcode=RetCode.EXCEPTION_ERROR, retmsg=str(e.args[0]), data=e.args[1])
    else:
        return get_json_result(retcode=RetCode.EXCEPTION_ERROR, retmsg=str(e))


def error_response(response_code, retmsg=None):
    if retmsg is None:
        retmsg = HTTP_STATUS_CODES.get(response_code, 'Unknown Error')
    return Response(json.dumps({'retmsg': retmsg, 'retcode': response_code}), status=response_code, mimetype='application/json')


def federated_api(job_id, method, endpoint, src_party_id, dest_party_id, src_role, json_body, federated_mode, api_version=API_VERSION,
                  overall_timeout=None):
    overall_timeout = JobDefaultConfig.remote_request_timeout if overall_timeout is None else overall_timeout
    if int(dest_party_id) == 0:
        federated_mode = FederatedMode.SINGLE
    if federated_mode == FederatedMode.SINGLE:
        return local_api(job_id=job_id, method=method, endpoint=endpoint, json_body=json_body, api_version=api_version)
    elif federated_mode == FederatedMode.MULTIPLE:
        host, port, protocol = get_federated_proxy_address(src_party_id, dest_party_id)
        if protocol == CoordinationCommunicationProtocol.HTTP:
            return federated_coordination_on_http(job_id=job_id, method=method, host=host,
                                                  port=port, endpoint=endpoint, src_party_id=src_party_id, src_role=src_role,
                                                  dest_party_id=dest_party_id, json_body=json_body, api_version=api_version, overall_timeout=overall_timeout)
        elif protocol == CoordinationCommunicationProtocol.GRPC:
            return federated_coordination_on_grpc(job_id=job_id, method=method, host=host,
                                                  port=port, endpoint=endpoint, src_party_id=src_party_id, src_role=src_role,
                                                  dest_party_id=dest_party_id, json_body=json_body, api_version=api_version, overall_timeout=overall_timeout)
        else:
            raise Exception(f"{protocol} coordination communication protocol is not supported.")
    else:
        raise Exception('{} work mode is not supported'.format(federated_mode))


def local_api(job_id, method, endpoint, json_body, api_version=API_VERSION, try_times=3):
    return federated_coordination_on_http(job_id=job_id, method=method, host=RuntimeConfig.JOB_SERVER_HOST,
                                          port=RuntimeConfig.HTTP_PORT, endpoint=endpoint, src_party_id="", src_role="",
                                          dest_party_id="", json_body=json_body, api_version=api_version, try_times=try_times)


def get_federated_proxy_address(src_party_id, dest_party_id):
    if isinstance(PROXY, str):
        if PROXY == CoordinationProxyService.ROLLSITE:
            proxy_address = ServiceRegistry.FATE_ON_EGGROLL.get(PROXY)
            return proxy_address["host"], proxy_address.get("grpc_port", proxy_address["port"]), CoordinationCommunicationProtocol.GRPC
        elif PROXY == CoordinationProxyService.NGINX:
            proxy_address = ServiceRegistry.FATE_ON_SPARK.get(PROXY)
            protocol = CoordinationCommunicationProtocol.HTTP if PROXY_PROTOCOL == "default" else PROXY_PROTOCOL
            return proxy_address["host"], proxy_address[f"{protocol}_port"], protocol
        else:
            raise RuntimeError(f"can not support coordinate proxy {PROXY}")
    elif isinstance(PROXY, dict):
        proxy_address = PROXY
        protocol = CoordinationCommunicationProtocol.HTTP if PROXY_PROTOCOL == "default" else PROXY_PROTOCOL
        proxy_name = PROXY.get("name", CoordinationProxyService.FATEFLOW)
        if proxy_name == CoordinationProxyService.FATEFLOW and str(dest_party_id) == str(src_party_id):
            host = RuntimeConfig.JOB_SERVER_HOST
            port = RuntimeConfig.HTTP_PORT
        else:
            host = proxy_address["host"]
            port = proxy_address[f"{protocol}_port"]
        return host, port, protocol
    else:
        raise RuntimeError(f"can not support coordinate proxy config {PROXY}")


def federated_coordination_on_http(job_id, method, host, port, endpoint, src_party_id, src_role, dest_party_id, json_body, api_version=API_VERSION, overall_timeout=None, try_times=3):
    overall_timeout = JobDefaultConfig.remote_request_timeout if overall_timeout is None else overall_timeout
    endpoint = f"/{api_version}{endpoint}"
    exception = None
    json_body['src_role'] = src_role
    json_body['src_party_id'] = src_party_id
    for t in range(try_times):
        try:
            url = "http://{}:{}{}".format(host, port, endpoint)
            audit_logger(job_id).info('remote http api request: {}'.format(url))
            action = getattr(requests, method.lower(), None)
            headers = HEADERS.copy()
            headers["dest-party-id"] = str(dest_party_id)
            headers["src-party-id"] = str(src_party_id)
            headers["src-role"] = str(src_role)
            http_response = action(url=url, data=json_dumps(json_body), headers=headers)
            audit_logger(job_id).info(http_response.text)
            response = http_response.json()
            audit_logger(job_id).info('remote http api response: {} {}'.format(endpoint, response))
            return response
        except Exception as e:
            exception = e
            schedule_logger(job_id).warning(f"remote http request {endpoint} error, sleep and try again")
            time.sleep(2 * (t+1))
    else:
        raise exception


def federated_coordination_on_grpc(job_id, method, host, port, endpoint, src_party_id, src_role, dest_party_id, json_body, api_version=API_VERSION,
                                   overall_timeout=None, try_times=3):
    overall_timeout = JobDefaultConfig.remote_request_timeout if overall_timeout is None else overall_timeout
    endpoint = f"/{api_version}{endpoint}"
    json_body['src_role'] = src_role
    json_body['src_party_id'] = src_party_id
    if CHECK_NODES_IDENTITY:
        get_node_identity(json_body, src_party_id)
    _packet = wrap_grpc_packet(json_body, method, endpoint, src_party_id, dest_party_id, job_id,
                               overall_timeout=overall_timeout)
    _routing_metadata = gen_routing_metadata(src_party_id=src_party_id, dest_party_id=dest_party_id)
    exception = None
    for t in range(try_times):
        try:
            channel, stub = get_command_federation_channel(host, port)
            _return, _call = stub.unaryCall.with_call(_packet, metadata=_routing_metadata, timeout=(overall_timeout/1000))
            audit_logger(job_id).info("grpc api response: {}".format(_return))
            channel.close()
            response = json_loads(_return.body.value)
            return response
        except Exception as e:
            exception = e
            schedule_logger(job_id).warning(f"remote request {endpoint} error, sleep and try again")
            time.sleep(2 * (t+1))
    else:
        tips = 'Please check rollSite and fateflow network connectivity'
        """
        if 'Error received from peer' in str(exception):
            tips = 'Please check if the fate flow server of the other party is started. '
        if 'failed to connect to all addresses' in str(exception):
            tips = 'Please check whether the rollsite service(port: 9370) is started. '
        """
        raise Exception('{}rpc request error: {}'.format(tips, exception))


def proxy_api(role, _job_id, request_config):
    job_id = request_config.get('header').get('job_id', _job_id)
    method = request_config.get('header').get('method', 'POST')
    endpoint = request_config.get('header').get('endpoint')
    src_party_id = request_config.get('header').get('src_party_id')
    dest_party_id = request_config.get('header').get('dest_party_id')
    json_body = request_config.get('body')
    _packet = forward_grpc_packet(json_body, method, endpoint, src_party_id, dest_party_id, job_id=job_id, role=role)
    _routing_metadata = gen_routing_metadata(src_party_id=src_party_id, dest_party_id=dest_party_id)
    host, port, protocol = get_federated_proxy_address(src_party_id, dest_party_id)
    channel, stub = get_command_federation_channel(host, port)
    _return, _call = stub.unaryCall.with_call(_packet, metadata=_routing_metadata)
    channel.close()
    json_body = json_loads(_return.body.value)
    return json_body


def forward_api(role, request_config):
    endpoint = request_config.get('header', {}).get('endpoint')
    url = "http://{}:{}{}".format(HOST, HTTP_PORT, endpoint)
    method = request_config.get('header', {}).get('method', 'post')
    audit_logger().info('api request: {}'.format(url))
    action = getattr(requests, method.lower(), None)
    http_response = action(url=url, json=request_config.get('body'), headers=HEADERS)
    response = http_response.json()
    audit_logger().info(response)
    return response


def get_node_identity(json_body, src_party_id):
    params = {
        'partyId': int(src_party_id),
        'federatedId': ServiceRegistry.FATEMANAGER.get("federatedId"),
    }
    try:
        response = requests.post(url="http://{}:{}{}".format(
            ServiceRegistry.FATEMANAGER.get("host"),
            ServiceRegistry.FATEMANAGER.get("port"),
            FATE_MANAGER_GET_NODE_INFO_ENDPOINT), json=params)
        json_body['appKey'] = response.json().get('data').get('appKey')
        json_body['appSecret'] = response.json().get('data').get('appSecret')
        json_body['_src_role'] = response.json().get('data').get('role')
    except Exception as e:
        raise Exception('get appkey and secret failed: {}'.format(str(e)))
