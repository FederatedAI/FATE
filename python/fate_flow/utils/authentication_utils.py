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
import functools
import json
import os

from flask import request

from fate_arch.common import file_utils
from fate_flow.settings import USE_AUTHENTICATION, PRIVILEGE_COMMAND_WHITELIST, stat_logger


class PrivilegeAuth(object):
    privilege_cache = {}
    local_storage_file = ''
    USE_LOCAL_STORAGE = True
    ALL_PERMISSION = {'privilege_role': ['guest', 'host', 'arbiter'],
                      'privilege_command': ['create', 'stop', 'run'],
                      'privilege_component': []}
    command_whitelist = None

    @classmethod
    def authentication_privilege(cls, src_party_id, src_role, request_path, party_id_index, role_index, command):
        if not src_party_id:
            return
        src_party_id = str(src_party_id)
        if src_party_id == PrivilegeAuth.get_dest_party_id(request_path, party_id_index):
            return
        stat_logger.info("party {} role {} start authentication".format(src_party_id, src_role))
        privilege_dic = PrivilegeAuth.get_authentication_items(request_path, role_index, command)
        for privilege_type, value in privilege_dic.items():
            if value in PrivilegeAuth.command_whitelist:
                continue
            if value:
                if value not in PrivilegeAuth.privilege_cache.get(src_party_id, {}).get(src_role, {}) \
                        .get(privilege_type, []):
                    if value not in PrivilegeAuth.get_permission_config(src_party_id, src_role) \
                            .get(privilege_type, []):
                        stat_logger.info('{} {} not authorized'.format(privilege_type.split('_')[1], value))
                        raise Exception('{} {} not authorized'.format(privilege_type.split('_')[1], value))
        stat_logger.info('party {} role {} successful authenticated'.format(src_party_id, src_role))
        return True

    @classmethod
    def get_permission_config(cls, src_party_id, src_role, use_local=True):
        if PrivilegeAuth.USE_LOCAL_STORAGE:
            return PrivilegeAuth.read_local_storage(src_party_id, src_role)
        else:
            return PrivilegeAuth.read_cloud_config_center(src_party_id, src_role)

    @classmethod
    def read_local_storage(cls, src_party_id, src_role):
        with open(PrivilegeAuth.local_storage_file) as fp:
            local_storage_conf = json.load(fp)
            PrivilegeAuth.privilege_cache = local_storage_conf
        return local_storage_conf.get(src_party_id, {}).get(src_role, {})

    @classmethod
    def get_new_permission_config(cls, src_party_id, src_role, privilege_role, privilege_command, privilege_component, delete):
        with open(PrivilegeAuth.local_storage_file) as f:
            stat_logger.info(
                "add permissions: src_party_id {} src_role {} privilege_role {} privilege_command {} privilege_component {}".format(
                    src_party_id, src_role, privilege_role, privilege_command, privilege_component))
            json_data = json.load(f)
            local_storage = json_data
            privilege_dic = PrivilegeAuth.get_privilege_dic(privilege_role, privilege_command, privilege_component)
            for privilege_type, values in privilege_dic.items():
                if values:
                    for value in values.split(','):
                        value = value.strip()
                        if value == 'all':
                            value = PrivilegeAuth.ALL_PERMISSION[privilege_type]
                        if value not in PrivilegeAuth.ALL_PERMISSION.get(privilege_type) and \
                                value != PrivilegeAuth.ALL_PERMISSION.get(privilege_type):
                            stat_logger.info('{} does not exist in the permission {}'.format(value, privilege_type.split('_')[1]))
                            raise Exception('{} does not exist in the permission {}'.format(value, privilege_type.split('_')[1]))
                        if not delete:
                            if local_storage.get(src_party_id, {}):
                                if local_storage.get(src_party_id).get(src_role, {}):
                                    if local_storage.get(src_party_id).get(src_role).get(privilege_type):
                                        local_storage[src_party_id][src_role][privilege_type].append(value) \
                                            if isinstance(value, str) \
                                            else local_storage[src_party_id][src_role][privilege_type].extend(value)
                                    else:
                                        local_storage[src_party_id][src_role][privilege_type] = [value] \
                                            if isinstance(value, str) \
                                            else value
                                else:
                                    local_storage[src_party_id][src_role] = {privilege_type: [value]} \
                                        if isinstance(value, str) \
                                        else {privilege_type: value}
                            else:
                                local_storage[src_party_id] = {src_role: {privilege_type: [value]}} \
                                    if isinstance(value, str)\
                                    else {src_role: {privilege_type: value}}
                            local_storage[src_party_id][src_role][privilege_type] = \
                                list(set(local_storage[src_party_id][src_role][privilege_type]))
                        else:
                            try:
                                if isinstance(value, str):
                                    local_storage[src_party_id][src_role][privilege_type].remove(value)
                                else:
                                    local_storage[src_party_id][src_role][ privilege_type] = []
                            except:
                                stat_logger.info('{} {} is not authorized ,it cannot be deleted'.format(privilege_type.split('_')[1], value)
                                    if isinstance(value, str) else "No permission to delete")
                                raise Exception(
                                    '{} {} is not authorized ,it cannot be deleted'.format(privilege_type.split('_')[1], value)
                                    if isinstance(value, str) else "No permission to delete")
            stat_logger.info('add permission successfully')
            f.close()
            return local_storage

    @classmethod
    def rewrite_local_storage(cls, new_json):
        with open(PrivilegeAuth.local_storage_file, 'w') as fp:
            PrivilegeAuth.privilege_cache = new_json
            json.dump(new_json, fp, indent=4, separators=(',', ': '))
            fp.close()

    @classmethod
    def read_cloud_config_center(cls, src_party_id, src_role):
        pass

    @classmethod
    def write_cloud_config_center(cls, ):
        pass

    @classmethod
    def get_authentication_items(cls, request_path, role_index, command):
        dest_role = request_path.split('/')[role_index]
        component = request.json.get('module_name').lower() if command == 'run' else None
        return PrivilegeAuth.get_privilege_dic(dest_role, command, component)

    @classmethod
    def get_dest_party_id(cls, request_path, party_id_index):
        return request_path.split('/')[party_id_index]

    @classmethod
    def get_privilege_dic(cls, privilege_role, privilege_command, privilege_component):
        return {'privilege_role': privilege_role,
                'privilege_command': privilege_command,
                'privilege_component': privilege_component}

    @classmethod
    def init(cls):
        if USE_AUTHENTICATION:
            # init local storage
            stat_logger.info('init local authorization library')
            file_dir = os.path.join(file_utils.get_python_base_directory(), 'fate_flow')
            os.makedirs(file_dir, exist_ok=True)
            PrivilegeAuth.local_storage_file = os.path.join(file_dir, 'authorization_config.json')
            if not os.path.exists(PrivilegeAuth.local_storage_file):
                with open(PrivilegeAuth.local_storage_file, 'w') as fp:
                    fp.write(json.dumps({}))

            # init whitelist
            PrivilegeAuth.command_whitelist = PRIVILEGE_COMMAND_WHITELIST

            # init ALL_PERMISSION
            component_path = os.path.join(file_utils.get_python_base_directory(),
                                          'federatedml', 'conf', 'setting_conf')
            search_command()
            stat_logger.info('search components from {}'.format(component_path))
            search_component(component_path)


def modify_permission(permission_info, delete=False):
    if PrivilegeAuth.USE_LOCAL_STORAGE:
        new_json = PrivilegeAuth.get_new_permission_config(src_party_id=permission_info.get('src_party_id'),
                                                           src_role=permission_info.get('src_role'),
                                                           privilege_role=permission_info.get('privilege_role', None),
                                                           privilege_command=permission_info.get('privilege_command',
                                                                                                 None),
                                                           privilege_component=permission_info.get(
                                                               'privilege_component', None),
                                                           delete=delete)
        PrivilegeAuth.rewrite_local_storage(new_json)


def request_authority_certification(party_id_index, role_index, command):
    def request_authority_certification_do(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if USE_AUTHENTICATION:
                PrivilegeAuth.authentication_privilege(src_party_id=request.json.get('src_party_id'),
                                                       src_role=request.json.get('src_role'),
                                                       request_path=request.path,
                                                       party_id_index=party_id_index,
                                                       role_index=role_index,
                                                       command=command
                                                       )
            return func(*args, **kwargs)
        return _wrapper
    return request_authority_certification_do


def search_command():
    command_list = []
    PrivilegeAuth.ALL_PERMISSION['privilege_command'].extend(command_list)


def search_component(path):
    component_list = [file_name.split('.')[0].lower() for file_name in os.listdir(path) if 'json' in file_name]
    component_list = list(set(component_list) - {'upload', 'download'})
    PrivilegeAuth.ALL_PERMISSION['privilege_component'].extend(component_list)


def authentication_check(src_role, src_party_id, dsl, runtime_conf, role, party_id):
    initiator = runtime_conf['initiator']
    roles = runtime_conf['role']
    if initiator['role'] != src_role or initiator['party_id'] != int(src_party_id) or int(party_id) not in roles[role]:
        if not int(party_id):
            return
        else:
            stat_logger.info('src_role {} src_party_id {} authentication check failed'.format(src_role, src_party_id))
            raise Exception('src_role {} src_party_id {} authentication check failed'.format(src_role, src_party_id))
    components = get_all_components(dsl)
    if str(party_id) == str(src_party_id):
        return
    need_run_commond = list(set(PrivilegeAuth.ALL_PERMISSION['privilege_command'])-set(PrivilegeAuth.command_whitelist))
    if need_run_commond != PrivilegeAuth.privilege_cache.get(src_party_id, {}).get(src_role, {}).get('privilege_command', []):
        if need_run_commond != PrivilegeAuth.get_permission_config(src_party_id, src_role).get('privilege_command', []):
            stat_logger.info('src_role {} src_party_id {} commond authentication that needs to be run failed:{}'.format(
                    src_role, src_party_id, set(need_run_commond) - set(PrivilegeAuth.privilege_cache.get(src_party_id,
                        {}).get(src_role, {}).get('privilege_command', []))))
            raise Exception('src_role {} src_party_id {} commond authentication that needs to be run failed:{}'.format(
                    src_role, src_party_id, set(need_run_commond) - set(PrivilegeAuth.privilege_cache.get(src_party_id,
                        {}).get(src_role, {}).get('privilege_command', []))))
    if not set(components).issubset(PrivilegeAuth.privilege_cache.get(src_party_id, {}).get(src_role, {}).get(
            'privilege_component', [])):
        if not set(components).issubset(PrivilegeAuth.get_permission_config(src_party_id, src_role).get(
                'privilege_component', [])):
            stat_logger.info('src_role {} src_party_id {} component authentication that needs to be run failed:{}'.format(
                src_role, src_party_id, components))
            raise Exception('src_role {} src_party_id {} component authentication that needs to be run failed:{}'.format(
                src_role, src_party_id, set(components)-set(PrivilegeAuth.privilege_cache.get(src_party_id, {}).get(
                    src_role, {}).get('privilege_component', []))))
    stat_logger.info('src_role {} src_party_id {} authentication check success'.format(src_role, src_party_id))


def check_constraint(job_runtime_conf, job_dsl):
    # Component constraint
    check_component_constraint(job_runtime_conf, job_dsl)


def check_component_constraint(job_runtime_conf, job_dsl):
    all_components = get_all_components(job_dsl)
    if 'glm' in all_components:
        roles = job_runtime_conf.get('role')
        if 'guest' in roles.keys() and 'arbiter' in roles.keys() and 'host' in roles.keys():
            for party_id in set(roles['guest']) & set(roles['arbiter']):
                if party_id in job_runtime_conf['host']:
                    raise Exception("GLM component constraint party id, please check role config")


def get_all_components(dsl):
    return [dsl['components'][component_name]['module'].lower() for component_name in dsl['components'].keys()]