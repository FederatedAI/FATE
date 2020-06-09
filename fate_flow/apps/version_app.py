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
from fate_flow.settings import stat_logger
from flask import Flask, request

from arch.api.utils.file_utils import get_fate_env, set_server_conf
from fate_flow.utils.api_utils import get_json_result


manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/get', methods=['POST'])
def get_fate_version_info():
    module, version = get_fate_env(request.json.get('module'))
    return get_json_result(data={module: version})


@manager.route('/set', methods=['POST'])
def set_fate_server_info():
    data = set_server_conf(request.json)
    return get_json_result(data=data)

