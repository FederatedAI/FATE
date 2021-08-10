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

from fate_arch.common import conf_utils
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.settings import stat_logger
from fate_flow.utils.api_utils import get_json_result

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/get', methods=['POST'])
def get_fate_version_info():
    version = RuntimeConfig.get_env(request.json.get('module', 'FATE'))
    return get_json_result(data={request.json.get('module'): version})


@manager.route('/registry', methods=['POST'])
def service_registry():
    update_server = {}
    service_config = request.json
    registry_service_list = ["fatemanager", "studio"]
    for k, v in service_config.items():
        if k not in registry_service_list:
            continue
        manager_conf = conf_utils.get_base_config(k, {})
        if not manager_conf:
            manager_conf = v
        manager_conf.update(v)
        conf_utils.update_config(k, manager_conf)
        update_server[k] = manager_conf
    return get_json_result(data={"update_server": update_server})
