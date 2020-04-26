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
import os
import shutil

from arch.api.utils.dtable_utils import get_table_info

from arch.api.utils.version_control import version_history, save_version
from flask import Flask, request, send_file

from fate_flow.settings import stat_logger, API_VERSION, SERVING_PATH, SERVINGS_ZK_PATH, \
    USE_CONFIGURATION_CENTER, ZOOKEEPER_HOSTS, SERVER_CONF_PATH, DEFAULT_MODEL_STORE_ADDRESS, TEMP_DIRECTORY
from fate_flow.driver.job_controller import JobController
from fate_flow.manager.model_manager import publish_model
from fate_flow.manager.model_manager import pipelined_model
from fate_flow.utils.api_utils import get_json_result, federated_api
from fate_flow.utils.job_utils import generate_job_id, runtime_conf_basic
from fate_flow.utils.node_check_utils import check_nodes
from fate_flow.utils.setting_utils import CenterConfig
from fate_flow.utils.detect_utils import check_config
from fate_flow.entity.constant_config import ModelOperation

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/load', methods=['POST'])
def load_model():
    request_config = request.json
    _job_id = generate_job_id()
    initiator_party_id = request_config['initiator']['party_id']
    initiator_role = request_config['initiator']['role']
    publish_model.generate_publish_model_info(request_config)
    load_status = True
    load_status_info = {}
    load_status_msg = 'success'
    for role_name, role_partys in request_config.get("role").items():
        if role_name == 'arbiter':
            continue
        load_status_info[role_name] = load_status_info.get(role_name, {})
        for _party_id in role_partys:
            request_config['local'] = {'role': role_name, 'party_id': _party_id}
            try:
                response = federated_api(job_id=_job_id,
                                         method='POST',
                                         endpoint='/{}/model/load/do'.format(API_VERSION),
                                         src_party_id=initiator_party_id,
                                         dest_party_id=_party_id,
                                         src_role = initiator_role,
                                         json_body=request_config,
                                         work_mode=request_config['job_parameters']['work_mode'])
                load_status_info[role_name][_party_id] = response['retcode']
            except Exception as e:
                stat_logger.exception(e)
                load_status = False
                load_status_msg = 'failed'
                load_status_info[role_name][_party_id] = 100
    return get_json_result(job_id=_job_id, retcode=(0 if load_status else 101), retmsg=load_status_msg,
                           data=load_status_info)


@manager.route('/load/do', methods=['POST'])
@check_nodes
def do_load_model():
    request_data = request.json
    request_data["servings"] = CenterConfig.get_settings(path=SERVING_PATH, servings_zk_path=SERVINGS_ZK_PATH,
                                                         use_zk=USE_CONFIGURATION_CENTER, hosts=ZOOKEEPER_HOSTS,
                                                         server_conf_path=SERVER_CONF_PATH)
    load_status = publish_model.load_model(config_data=request_data)
    return get_json_result(retcode=(0 if load_status else 101))


@manager.route('/bind', methods=['POST'])
def bind_model_service():
    request_config = request.json
    if not request_config.get('servings'):
        # get my party all servings
        request_config['servings'] = CenterConfig.get_settings(path=SERVING_PATH, servings_zk_path=SERVINGS_ZK_PATH,
                                                               use_zk=USE_CONFIGURATION_CENTER, hosts=ZOOKEEPER_HOSTS,
                                                               server_conf_path=SERVER_CONF_PATH)
    if not request_config.get('service_id'):
        return get_json_result(retcode=101, retmsg='no service id')
    bind_status, service_id = publish_model.bind_model_service(config_data=request_config)
    return get_json_result(retcode=(0 if bind_status else 101), retmsg='service id is {}'.format(service_id))


@manager.route('/transfer', methods=['post'])
def transfer_model():
    model_data = publish_model.download_model(request.json)
    return get_json_result(retcode=0, retmsg="success", data=model_data)


@manager.route('/version/save', methods=['POST'])
def save_table_version():
    save_version(name=request.json.get('name'),
                 namespace=request.json.get('namespace'),
                 version_log=request.json.get('version_log'))
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/table_info', methods=['POST'])
def get_table():
    table_name, namespace = get_table_info(config=request.json.get('config'), create=request.json.get('create'))
    return get_json_result(retcode=0, retmsg='success', data={'table_name':table_name, 'namespace': namespace})


@manager.route('/version_history', methods=['POST'])
def query_model_version_history():
    history = version_history(data_table_namespace=request.json.get("namespace"))
    return get_json_result(data=history)


@manager.route('/<model_operation>', methods=['post', 'get'])
def operate_model(model_operation):
    request_config = request.json or request.form.to_dict()
    job_id = generate_job_id()
    required_arguments = ["model_id", "model_version"]
    if model_operation not in [ModelOperation.STORE, ModelOperation.RESTORE, ModelOperation.EXPORT, ModelOperation.IMPORT]:
        raise Exception('Can not support this operating now: {}'.format(model_operation))
    check_config(request_config, required_arguments=required_arguments)
    if model_operation in [ModelOperation.EXPORT, ModelOperation.IMPORT]:
        if model_operation == ModelOperation.IMPORT:
            file = request.files.get('file')
            file_path = os.path.join(TEMP_DIRECTORY, file.filename)
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                file.save(file_path)
            except Exception as e:
                shutil.rmtree(file_path)
                raise e
            request_config['file'] = file_path
            model = pipelined_model.PipelinedModel(model_id=request_config["model_id"], model_version=request_config["model_version"])
            model.unpack_model(file_path)
            return get_json_result()
        else:
            model = pipelined_model.PipelinedModel(model_id=request_config["model_id"], model_version=request_config["model_version"])
            archive_file_path = model.packaging_model()
            return send_file(archive_file_path, attachment_filename=os.path.basename(archive_file_path), as_attachment=True)
    else:
        data = {}
        job_dsl, job_runtime_conf = gen_model_operation_job_config(request_config, model_operation)
        job_id, job_dsl_path, job_runtime_conf_path, logs_directory, model_info, board_url = JobController.submit_job(
            {'job_dsl': job_dsl, 'job_runtime_conf': job_runtime_conf}, job_id=job_id)
        data.update({'job_dsl_path': job_dsl_path, 'job_runtime_conf_path': job_runtime_conf_path,
                     'board_url': board_url, 'logs_directory': logs_directory})
        return get_json_result(job_id=job_id, data=data)


def gen_model_operation_job_config(config_data: dict, model_operation: ModelOperation):
    job_runtime_conf = runtime_conf_basic(if_local=True)
    initiator_role = "local"
    job_dsl = {
        "components": {}
    }

    if model_operation in [ModelOperation.STORE, ModelOperation.RESTORE]:
        component_name = "{}_0".format(model_operation)
        component_parameters = dict()
        component_parameters.update(config_data)
        for k, v in config_data.items():
            component_parameters[k] = [v]
        if "store_address" not in config_data:
            component_parameters["store_address"] = [DEFAULT_MODEL_STORE_ADDRESS]
        job_runtime_conf["role_parameters"][initiator_role] = {component_name: component_parameters}
        job_dsl["components"][component_name] = {
            "module": "Model{}".format(model_operation.capitalize())
        }
    else:
        raise Exception("Can not support this model operation: {}".format(model_operation))
    return job_dsl, job_runtime_conf