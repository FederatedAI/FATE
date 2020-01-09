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
import base64

import grpc

from arch.api import session
from arch.api.model_manager.manager import get_model_table_partition_count
from arch.api.proto import model_service_pb2
from arch.api.proto import model_service_pb2_grpc
from arch.api.utils.core import get_fate_uuid
from fate_flow.settings import stat_logger
from fate_flow.utils import model_utils


def generate_publish_model_info(config_data):
    model_id = config_data['job_parameters']['model_id']
    model_version = config_data['job_parameters']['model_version']
    config_data['model'] = {}
    for role, role_party in config_data.get("role").items():
        config_data['model'][role] = {}
        for party_id in role_party:
            config_data['model'][role][party_id] = {
                'model_id': model_utils.gen_party_model_id(model_id, role, party_id),
                'model_version': model_version}


def load_model(config_data):
    stat_logger.info(config_data)
    success = True
    for serving in config_data.get('servings'):
        with grpc.insecure_channel(serving) as channel:
            stub = model_service_pb2_grpc.ModelServiceStub(channel)
            load_model_request = model_service_pb2.PublishRequest()
            for role_name, role_partys in config_data.get("role").items():
                for _party_id in role_partys:
                    load_model_request.role[role_name].partyId.append(_party_id)
            for role_name, role_model_config in config_data.get("model").items():
                for _party_id, role_party_model_config in role_model_config.items():
                    load_model_request.model[role_name].roleModelInfo[_party_id].tableName = role_party_model_config[
                        'model_version']
                    load_model_request.model[role_name].roleModelInfo[_party_id].namespace = role_party_model_config[
                        'model_id']
            stat_logger.info('request serving: {} load model'.format(serving))
            load_model_request.local.role = config_data.get('local').get('role')
            load_model_request.local.partyId = config_data.get('local').get('party_id')
            stat_logger.info(load_model_request)
            response = stub.publishLoad(load_model_request)
            stat_logger.info(
                '{} {} load model status: {}'.format(load_model_request.local.role, load_model_request.local.partyId,
                                                     response.statusCode))
            if response.statusCode != 0:
                success = False
    return success


def bind_model_service(config_data):
    service_id = config_data.get('service_id')
    initiator_role = config_data['initiator']['role']
    initiator_party_id = config_data['initiator']['party_id']
    model_id = config_data['job_parameters']['model_id']
    model_version = config_data['job_parameters']['model_version']
    status = True
    for serving in config_data.get('servings'):
        with grpc.insecure_channel(serving) as channel:
            stub = model_service_pb2_grpc.ModelServiceStub(channel)
            publish_model_request = model_service_pb2.PublishRequest()
            publish_model_request.serviceId = service_id
            for role_name, role_party in config_data.get("role").items():
                publish_model_request.role[role_name].partyId.extend(role_party)

            publish_model_request.model[initiator_role].roleModelInfo[initiator_party_id].tableName = model_version
            publish_model_request.model[initiator_role].roleModelInfo[
                initiator_party_id].namespace = model_utils.gen_party_model_id(model_id, initiator_role,
                                                                               initiator_party_id)
            publish_model_request.local.role = initiator_role
            publish_model_request.local.partyId = initiator_party_id
            stat_logger.info(publish_model_request)
            response = stub.publishBind(publish_model_request)
            stat_logger.info(response)
            if response.statusCode != 0:
                status = False
    return status, service_id


def download_model(request_data):
    pipeline_model_table = session.table(name=request_data.get('name'), namespace=request_data.get('namespace'),
                                         partition=get_model_table_partition_count(),
                                         create_if_missing=False, error_if_exist=False)
    model_data = {}
    for storage_key, buffer_object_bytes in pipeline_model_table.collect(use_serialize=False):
        model_data[storage_key] = base64.b64encode(buffer_object_bytes).decode()
    return model_data

