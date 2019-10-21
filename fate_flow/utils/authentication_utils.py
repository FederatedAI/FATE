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
import re

from flask import request

from arch.api.utils import file_utils
from fate_flow.settings import USE_AUTHENTICATION, PRIVILEGE_COMMAND_WHITELIST, stat_logger, \
    CLUSTER_STANDALONE_JOB_SERVER_PORT


class PrivilegeAuth(object):
    privilege_cache = {}
    local_storage_file = ''
    USE_LOCAL_STORAGE = True
    ALL_PERMISSION = {'privilege_role': ['guest', 'host', 'arbiter'],
                      'privilege_command': [],
                      'privilege_component': []}
    command_whitelist = None

    @staticmethod
    def authentication_privilege(src_party_id, src_role, request_path, func_name):
        if request.url_root.split(':')[-1].split('/')[0] == str(CLUSTER_STANDALONE_JOB_SERVER_PORT):
            return
        if not int(PrivilegeAuth.get_dest_party_id(request_path, func_name)) or \
                src_party_id == PrivilegeAuth.get_dest_party_id(request_path, func_name):
            return
        if src_role != 'guest':
            stat_logger.info('src_role {} is not guest'.format(src_role))
            raise Exception('src_role {} is not guest'.format(src_role))
        if src_party_id == PrivilegeAuth.get_dest_party_id(request_path, func_name):
            return
        stat_logger.info("party {} role {} start authentication".format(src_party_id, src_role))
        privilege_dic = PrivilegeAuth.get_authentication_items(request_path, func_name)
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

    @staticmethod
    def get_permission_config(src_party_id, src_role, use_local=True):
        if PrivilegeAuth.USE_LOCAL_STORAGE:
            return PrivilegeAuth.read_local_storage(src_party_id, src_role)
        else:
            return PrivilegeAuth.read_cloud_config_center(src_party_id, src_role)

    @staticmethod
    def read_local_storage(src_party_id, src_role):
        with open(PrivilegeAuth.local_storage_file) as fp:
            local_storage_conf = json.load(fp)
            PrivilegeAuth.privilege_cache = local_storage_conf
        return local_storage_conf.get(src_party_id, {}).get(src_role, {})

    @staticmethod
    def get_new_permission_config(src_party_id, src_role, privilege_role, privilege_command, privilege_component, delete):
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

    @staticmethod
    def rewrite_local_storage(new_json):
        with open(PrivilegeAuth.local_storage_file, 'w') as fp:
            PrivilegeAuth.privilege_cache = new_json
            json.dump(new_json, fp, indent=4, separators=(',', ': '))
            fp.close()

    @staticmethod
    def read_cloud_config_center(src_party_id, src_role):
        pass

    @staticmethod
    def write_cloud_config_center():
        pass

    @staticmethod
    def get_authentication_items(request_path, func_name):
        dest_role = request_path.split('/')[2] if 'task' not in func_name else request_path.split('/')[4]
        component = request.json.get('module_name').lower() if 'run_task' in func_name else None
        return PrivilegeAuth.get_privilege_dic(dest_role, func_name, component)

    @staticmethod
    def get_dest_party_id(request_path, func_name):
        return request_path.split('/')[3] if 'task' not in func_name else request_path.split('/')[5]

    @staticmethod
    def get_privilege_dic(privilege_role, privilege_command, privilege_component):
        return {'privilege_role': privilege_role,
                'privilege_command': privilege_command,
                'privilege_component': privilege_component}

    @staticmethod
    def init():
        if USE_AUTHENTICATION:
            # init local storage
            stat_logger.info('init local authorization library')
            file_dir = os.path.join(file_utils.get_project_base_directory(), 'fate_flow')
            os.makedirs(file_dir, exist_ok=True)
            PrivilegeAuth.local_storage_file = os.path.join(file_dir, 'authorization_config.json')
            if not os.path.exists(PrivilegeAuth.local_storage_file):
                with open(PrivilegeAuth.local_storage_file, 'w') as fp:
                    fp.write(json.dumps({}))

            # init whitelist
            PrivilegeAuth.command_whitelist = PRIVILEGE_COMMAND_WHITELIST

            # init ALL_PERMISSION
            component_path = os.path.join(file_utils.get_project_base_directory(),
                                          'federatedml', 'conf', 'setting_conf')
            command_file_path = os.path.join(file_utils.get_project_base_directory(),
                                             'fate_flow', 'apps', 'schedule_app.py')
            stat_logger.info('search commands from {}'.format(command_file_path))
            search_command(command_file_path)
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


def request_authority_certification(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        if USE_AUTHENTICATION:
            PrivilegeAuth.authentication_privilege(src_party_id=str(request.json.get('src_party_id')),
                                                   src_role=request.json.get('src_role'),
                                                   request_path=request.path,
                                                   func_name=func.__name__
                                                   )
        return func(*args, **kwargs)
    return _wrapper


def search_command(path):
    with open(path, 'r') as fp:
        command_list = re.findall("def (.*)\(", fp.read())
    command_list = list(set(command_list) - {'internal_server_error', 'kill_job', 'task_status', 'job_status', 'cancel_job'})
    PrivilegeAuth.ALL_PERMISSION['privilege_command'].extend(command_list)


def search_component(path):
    component_list = [file_name.split('.')[0].lower() for file_name in os.listdir(path) if 'json' in file_name]
    component_list = list(set(component_list) - {'upload', 'download'})
    PrivilegeAuth.ALL_PERMISSION['privilege_component'].extend(component_list)


def authentication_check(src_role, src_party_id, dsl, runtime_conf, role, party_id):
    initiator = runtime_conf['initiator']
    roles = runtime_conf['role']
    if request.url_root.split(':')[-1].split('/')[0] == str(CLUSTER_STANDALONE_JOB_SERVER_PORT):
        return
    if 'local' not in roles or str(party_id) != str(src_party_id):
        if set(roles['host']) & set(roles['guest']):
            stat_logger.info('host {} became guest'.format(set(roles['host']) & set(roles['guest'])))
            raise Exception('host {} can not be used as guest'.format(set(roles['host']) & set(roles['guest'])))
    if initiator['role'] != src_role or initiator['party_id'] != int(src_party_id) or int(party_id) not in roles[role]:
        if not int(party_id):
            return
        else:
            stat_logger.info('src_role {} src_party_id {} authentication check failed'.format(src_role, src_party_id))
            raise Exception('src_role {} src_party_id {} authentication check failed'.format(src_role, src_party_id))
    components = [dsl['components'][component_name]['module'].lower() for component_name in dsl['components'].keys()]
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