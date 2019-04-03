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
from arch.task_manager.utils.job_utils import generate_job_id
from flask import Flask, request
import grpc
from arch.task_manager.settings import server_conf
from arch.task_manager.utils import publish_model
from arch.task_manager.utils.grpc_utils import get_proxy_data_channel, wrap_grpc_packet
from arch.task_manager.utils.job_utils import get_json_result
from arch.api.version_control.control import version_history
from arch.api import eggroll
from arch.task_manager.settings import WORK_MODE, logger
import json
manager = Flask(__name__)


@manager.route('/load', methods=['POST'])
def load_model():
    config = file_utils.load_json_conf(request.json.get("config_path"))
    _job_id = generate_job_id()
    channel, stub = get_proxy_data_channel()
    for _party_id in config.get("party_ids"):
        config['my_party_id'] = _party_id
        _method = 'POST'
        _url = '/model/load/do'
        _packet = wrap_grpc_packet(config, _method, _url, _party_id, _job_id)
        logger.info(
            'Starting load model job_id:{} party_id:{} method:{} url:{}'.format(_job_id, _party_id,_method, _url))
        try:
            _return = stub.unaryCall(_packet)
            logger.info("Grpc unary response: {}".format(_return))
        except grpc.RpcError as e:
            msg = 'job_id:{} party_id:{} method:{} url:{} Failed to start load model'.format(_job_id,
                                                                                             _party_id,
                                                                                             _method,
                                                                                             _url)
            logger.exception(msg)
            return get_json_result(-101, 'UnaryCall submit to remote manager failed')


@manager.route('/load/do', methods=['POST'])
def do_load_model():
    request_data = request.json
    try:
        request_data["servings"] = server_conf.get("servers", {}).get("servings", [])
        publish_model.load_model(config_data=request_data)
        return get_json_result()
    except Exception as e:
        logger.exception(e)
        return get_json_result(status=1, msg="load model error: %s" % e)


@manager.route('/online', methods=['POST'])
def publish_model_online():
    request_data = request.json
    try:
        config = file_utils.load_json_conf(request_data.get("config_path"))
        publish_model.publish_online(config_data=config)
        return get_json_result()
    except Exception as e:
        logger.exception(e)
        return get_json_result(status=1, msg="publish model error: %s" % e)


@manager.route('/version', methods=['POST'])
def query_model_version_history():
    request_data = request.json
    try:
        config = file_utils.load_json_conf(request_data.get("config_path"))
        print(config)
        eggroll.init(mode=WORK_MODE)
        history = version_history(data_table_namespace=config.get("namespace"))
        return get_json_result(msg=json.dumps(history))
    except Exception as e:
        logger.exception(e)
        return get_json_result(status=1, msg="load model error: %s" % e)
