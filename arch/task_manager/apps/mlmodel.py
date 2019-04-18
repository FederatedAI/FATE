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
from arch.api.utils import file_utils
from flask import Flask, request
from arch.task_manager.settings import server_conf
from arch.task_manager.utils import publish_model
from arch.task_manager.job_manager import generate_job_id
from arch.task_manager.utils.api_utils import get_json_result, federated_api
from arch.api.version_control.control import version_history
from arch.api import eggroll
from arch.task_manager.settings import WORK_MODE, logger, SERVINGS
import json
manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    logger.exception(e)
    return get_json_result(100, str(e))


@manager.route('/load', methods=['POST'])
def load_model():
    config = file_utils.load_json_conf(request.json.get("config_path"))
    _job_id = generate_job_id()
    for _party_id in config.get("party_ids"):
        st, msg = federated_api(job_id=_job_id,
                                method='POST',
                                url='/model/load/do',
                                party_id=_party_id,
                                data=config)
    return get_json_result()


@manager.route('/load/do', methods=['POST'])
def do_load_model():
    request_data = request.json
    request_data["servings"] = server_conf.get("servers", {}).get("servings", [])
    publish_model.load_model(config_data=request_data)
    return get_json_result()


@manager.route('/online', methods=['POST'])
def publish_model_online():
    request_data = request.json
    config = file_utils.load_json_conf(request_data.get("config_path"))
    if not config.get('servings'):
        # get my party all servings
        config['servings'] = SERVINGS
    publish_model.publish_online(config_data=config)
    return get_json_result()


@manager.route('/version', methods=['POST'])
def query_model_version_history():
    request_data = request.json
    config = file_utils.load_json_conf(request_data.get("config_path"))
    eggroll.init(mode=WORK_MODE)
    history = version_history(data_table_namespace=config.get("namespace"))
    return get_json_result(msg=json.dumps(history))
