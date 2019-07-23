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
import grpc
from arch.api.proto import model_service_pb2
from arch.api.proto import model_service_pb2_grpc
from arch.api.utils import dtable_utils
import copy
from fate_flow.settings import stat_logger


def generate_publish_model_info(config_data):
    default_table_config = dict()
    default_table_config['role'] = config_data.get('role')
    default_table_config['data_type'] = 'model'
    default_table_config['gen_table_info'] = config_data.get('gen_table_info', False)
    table_config = copy.deepcopy(default_table_config)

    # The model version is consistent under the same modeling, first find the model version from initiator
    table_config['local'] = config_data['initiator']
    initiator_role = table_config['local']['role']
    initiator_party_id = table_config['local']['party_id']
    initiator_model_id = config_data['model'][initiator_role][initiator_party_id]['model_id']
    initiator_model_version = config_data['model'][initiator_role][initiator_party_id]['model_version']
    table_config.update({'table_name': initiator_model_version, 'namespace': initiator_model_id})
    model_version, model_id = dtable_utils.get_table_info(config=table_config)
    # get model version
    models_version = model_version
    if not models_version or not model_id:
        return False
    for role_name, role_model_config in config_data.get("model").items():
        for _party_id, role_party_model_config in role_model_config.items():
            table_config = copy.deepcopy(default_table_config)
            table_config['local'] = {'role': role_name, 'party_id': _party_id}
            table_config.update(role_party_model_config)
            table_config['namespace'] = role_party_model_config.get('model_id', '')
            table_config['table_name'] = table_config['model_version'] if table_config.get('model_version') else models_version
            model_version, model_id = dtable_utils.get_table_info(config=table_config)
            config_data['model'][role_name][_party_id]['model_version'] = model_version
            config_data['model'][role_name][_party_id]['model_id'] = model_id
    return True


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
                    load_model_request.model[role_name].roleModelInfo[_party_id].tableName = role_party_model_config['model_version']
                    load_model_request.model[role_name].roleModelInfo[_party_id].namespace = role_party_model_config['model_id']
            stat_logger.info('request serving: {} load model'.format(serving))
            load_model_request.local.role = config_data.get('local').get('role')
            load_model_request.local.partyId = config_data.get('local').get('party_id')
            stat_logger.info(load_model_request)
            response = stub.publishLoad(load_model_request)
            stat_logger.info('{} {} load model status: {}'.format(load_model_request.local.role, load_model_request.local.partyId, response.statusCode))
            if response.statusCode != 0:
                success = False
    return success


def publish_online(config_data):
    _role = config_data.get('local').get('role')
    _party_id = config_data.get('local').get('party_id')
    success = True
    for serving in config_data.get('servings'):
        with grpc.insecure_channel(serving) as channel:
            stub = model_service_pb2_grpc.ModelServiceStub(channel)
            publish_model_request = model_service_pb2.PublishRequest()
            for role_name, role_party in config_data.get("role").items():
                publish_model_request.role[role_name].partyId.extend(role_party)

            for role_name, role_model_config in config_data.get("model").items():
                if role_name != _role:
                    continue
                if role_model_config.get(_party_id):
                    table_config = copy.deepcopy(role_model_config.get(_party_id))
                    table_config['local'] = {'role': _role, 'party_id': _party_id}
                    table_config['role'] = config_data.get('role')
                    table_config['data_type'] = 'model'
                    table_config['gen_table_info'] = True
                    table_config['namespace'] = table_config.get('model_id', '')
                    table_config['table_name'] = table_config.get('model_version', '')
                    table_name, namespace = dtable_utils.get_table_info(config=table_config)
                    publish_model_request.model[_role].roleModelInfo[_party_id].tableName = table_name
                    publish_model_request.model[_role].roleModelInfo[_party_id].namespace = namespace
            publish_model_request.local.role = _role
            publish_model_request.local.partyId = _party_id
            stat_logger.info(publish_model_request)
            response = stub.publishOnline(publish_model_request)
            stat_logger.info(response)
            if response.statusCode != 0:
                success = False
    return success
