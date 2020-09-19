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
from flask import Flask, request

from fate_flow.settings import stat_logger
from fate_flow.utils.api_utils import get_json_result
from fate_flow.utils.authentication_utils import modify_permission, PrivilegeAuth

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/grant/privilege', methods=['post'])
def grant_permission():
    modify_permission(request.json)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/delete/privilege', methods=['post'])
def delete_permission():
    modify_permission(request.json, delete=True)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/query/privilege', methods=['post'])
def query_privilege():
    privilege_dict = PrivilegeAuth.get_permission_config(request.json.get('src_party_id'), request.json.get('src_role'))
    return get_json_result(retcode=0, retmsg='success', data={'src_party_id': request.json.get('src_party_id'),
                                                              'role': request.json.get('src_role'),
                                                              'privilege_role': privilege_dict.get('privilege_role',[]),
                                                              'privilege_command': privilege_dict.get('privilege_command', []),
                                                              'privilege_component': privilege_dict.get('privilege_component', [])})