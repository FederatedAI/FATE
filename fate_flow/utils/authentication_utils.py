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

from arch.api.utils import file_utils
from fate_flow.settings import USE_AUTHENTICATION


class PrivilegeAuth(object):

    privilege_cache = {}
    file = ''
    ALL_PERMISSION = {'privilege_role': ['guest', 'host', 'arbiter'],
                      'privilege_command': ['run', 'stop', 'cancel'],
                      'privilege_component': ['data_io']}

    @staticmethod
    def authentication_privilege(src_party_id, src_role, request_path):
        dest_role, user_command, component = PrivilegeAuth.get_authentication_items(request_path)
        privilege_dic = PrivilegeAuth.get_privilege_dic(dest_role, user_command, component)
        for privilege_type, value in privilege_dic.items():
            if privilege_type:
                if privilege_type not in PrivilegeAuth.privilege_cache.get(src_party_id, {}).get(src_role, {}) \
                        .get(privilege_type, []):
                    if privilege_type not in PrivilegeAuth.get_permission_config(src_party_id, src_role) \
                            .get(privilege_type, []):
                        raise Exception('{} {} not authorized'.format(privilege_type.split('_')[1], value))
                    else:
                        PrivilegeAuth.privilege_cache[src_party_id][src_role]['privilege_command'] = user_command
        return True

    @staticmethod
    def get_permission_config(src_party_id, src_role, use_local=True):
        return PrivilegeAuth.read_local_storage(src_party_id, src_role)

    @staticmethod
    def read_local_storage(src_party_id, src_role):
        with open(PrivilegeAuth.file) as fp:
            local_storage_conf = json.load(fp)
        return local_storage_conf.get(src_party_id, {}).get(src_role, {})

    @staticmethod
    def get_new_permission_config(src_party_id, src_role, privilege_role, privilege_command, privilege_component, delete):
        with open(PrivilegeAuth.file) as f:
            json_data = json.load(f)
            local_storage = json_data
            privilege_dic = PrivilegeAuth.get_privilege_dic(privilege_role, privilege_command, privilege_component)
            for privilege_type, values in privilege_dic.items():
                if values:
                    for value in values.split(','):
                        value = value.strip()
                        if value not in PrivilegeAuth.ALL_PERMISSION.get(privilege_type):
                            raise Exception(
                                '{} does not exist in the permission {}'.format(value, privilege_type.split('_')[1]))
                        if not delete:
                            if local_storage.get(src_party_id, {}):
                                if local_storage.get(src_party_id).get(src_role, {}):

                                    if local_storage.get(src_party_id).get(src_role).get(privilege_type):
                                        local_storage[src_party_id][src_role][privilege_type].append(value)
                                    else:
                                        local_storage[src_party_id][src_role][privilege_type] = [value]

                                else:
                                    local_storage[src_party_id][src_role] = {privilege_type: [value]}
                            else:
                                local_storage[src_party_id] = {src_role: {privilege_type: [value]}}
                            local_storage[src_party_id][src_role][privilege_type] = \
                                list(set(local_storage[src_party_id][src_role][privilege_type]))
                        else:
                            try:
                                local_storage[src_party_id][src_role][privilege_type].remove(value)
                            except:
                                raise Exception('{} {} is not authorized ,it cannot be deleted'.format(privilege_type.split('_')[1],value))
            f.close()
            return local_storage

    @staticmethod
    def rewrite_local_storage(new_json):
        with open(PrivilegeAuth.file, 'w') as fp:
            PrivilegeAuth.privilege_cache = new_json
            json.dump(new_json, fp, indent=4, separators=(',', ': '))
            fp.close()

    @staticmethod
    def read_cloud_config_center():
        pass

    @staticmethod
    def write_cloud_config_center():
        pass

    @staticmethod
    def get_authentication_items(request_path):
        current_operation = request_path.splite('/')[-1]
        dest_role = request_path.splite('/')[-3] if current_operation != 'pipeline' else request_path.splite('/')[-6]
        component = request_path.splite('/')[-5] if current_operation == 'run' else None
        command = None
        if current_operation == 'create':
            command = 'run'
        if current_operation == 'kill':
            command = 'stop'
        if current_operation == 'cancel':
            command = 'cancel'
        return command, dest_role, component

    @staticmethod
    def get_privilege_dic(privilege_role, privilege_command, privilege_component):
        return {'privilege_role': privilege_role,
                'privilege_command': privilege_command,
                'privilege_component': privilege_component}

    @staticmethod
    def init():
        if USE_AUTHENTICATION:
            PrivilegeAuth.file = os.path.join(file_utils.get_project_base_directory(), 'fate_flow',
                                              'authorization_config.json')
            if not os.path.exists(PrivilegeAuth.file):
                with open(PrivilegeAuth.file, 'w') as fp:
                    fp.write(json.dumps({}))


def modify_permission(permission_info, delete=False):
    new_json = PrivilegeAuth.get_new_permission_config(src_party_id=permission_info.get('src_party_id'),
                                                       src_role=permission_info.get('src_role'),
                                                       privilege_role=permission_info.get('privilege_role', None),
                                                       privilege_command=permission_info.get('privilege_role', None),
                                                       privilege_component=permission_info.get('privilege_role', None),
                                                       delete=delete)
    PrivilegeAuth.rewrite_local_storage(new_json)


def request_authority_certification(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        if USE_AUTHENTICATION:
            PrivilegeAuth.authentication_privilege(src_party_id=str(request.json.get('src_party_id')),
                                                   src_role=request.json.get('src_role'),
                                                   request_path=request.path
                                                   )
        return func(*args, **kwargs)
    return _wrapper












