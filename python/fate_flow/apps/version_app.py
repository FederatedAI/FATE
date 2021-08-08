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
from flask import request

from fate_arch.common import conf_utils
from fate_flow.runtime_config import RuntimeConfig
from fate_flow.utils.api_utils import get_json_result
from fate_flow.settings import ServiceSettings, JobDefaultSettings, API_VERSION


@manager.route('/get', methods=['POST'])
def get_fate_version_info():
    module = request.json.get('module', 'FATE')
    version = RuntimeConfig.get_env(module)
    return get_json_result(data={
        module: version,
        'API': API_VERSION,
    })


@manager.route('/set', methods=['POST'])
def set_fate_server_info():
    # manager
    federated_id = request.json.get("federatedId")
    ServiceSettings.FATEMANAGER["federatedId"] = federated_id
    conf_utils.update_config("fatemanager", ServiceSettings.FATEMANAGER)
    return get_json_result(data={"federatedId": federated_id})


@manager.route('/reload', methods=['POST'])
def reload_settings():
    settings = {
        "job_default_settings": JobDefaultSettings.load(),
        "service_settings": ServiceSettings.load()
    }
    return get_json_result(data=settings)
