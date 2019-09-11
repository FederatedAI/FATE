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

from flask import Flask, request

from arch.api.utils import file_utils
from fate_flow.settings import JOB_MODULE_CONF
from fate_flow.settings import stat_logger, CLUSTER_STANDALONE_JOB_SERVER_PORT
from fate_flow.utils.api_utils import get_json_result, request_execute_server
from fate_flow.utils.job_utils import generate_job_id, get_job_directory, new_runtime_conf, run_subprocess
from fate_flow.utils import detect_utils
<<<<<<< HEAD
=======
from fate_flow.driver.job_controller import JobController
>>>>>>> fate_flow: improve data access interface
from fate_flow.entity.constant_config import WorkMode
from fate_flow.entity.runtime_config import RuntimeConfig

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


<<<<<<< HEAD
@manager.route('/<data_func>', methods=['post'])
def download_upload(data_func):
    request_config = request.json
    _job_id = generate_job_id()
    stat_logger.info('generated job_id {}, body {}'.format(_job_id, request_config))
    _job_dir = get_job_directory(_job_id)
    os.makedirs(_job_dir, exist_ok=True)
    module = data_func
    required_arguments = ['work_mode', 'namespace', 'table_name']
    if module == 'upload':
        required_arguments.extend(['file', 'head', 'partition'])
    elif module == 'download':
        required_arguments.extend(['output_path'])
    else:
        raise Exception('can not support this operating: {}'.format(module))
    detect_utils.check_config(request_config, required_arguments=required_arguments)
    job_work_mode = request_config['work_mode']
    # todo: The current code here is redundant with job_app/submit_job, the next version of this function will be implemented by job_app/submit_job
    if job_work_mode != RuntimeConfig.WORK_MODE:
        if RuntimeConfig.WORK_MODE == WorkMode.CLUSTER and job_work_mode == WorkMode.STANDALONE:
            # use cluster standalone job server to execute standalone job
            return request_execute_server(request=request, execute_host='{}:{}'.format(request.remote_addr, CLUSTER_STANDALONE_JOB_SERVER_PORT))
        else:
            raise Exception('server run on standalone can not support cluster mode job')

    if module == "upload":
        if not os.path.isabs(request_config['file']):
            request_config["file"] = os.path.join(file_utils.get_project_base_directory(), request_config["file"])
    try:
        conf_file_path = new_runtime_conf(job_dir=_job_dir, method=data_func, module=module,
                                          role=request_config.get('local', {}).get("role"),
                                          party_id=request_config.get('local', {}).get("party_id", ''))
        file_utils.dump_json_conf(request_config, conf_file_path)
        progs = ["python3",
                 os.path.join(file_utils.get_project_base_directory(), JOB_MODULE_CONF[module]["module_path"]),
                 "-j", _job_id,
                 "-c", conf_file_path
                 ]
        try:
            p = run_subprocess(config_dir=_job_dir, process_cmd=progs)
        except Exception as e:
            stat_logger.exception(e)
            p = None
        return get_json_result(retcode=(0 if p else 101), job_id=_job_id,
                               data={'table_name': request_config['table_name'],
                                     'namespace': request_config['namespace'], 'pid': p.pid if p else ''})
    except Exception as e:
        stat_logger.exception(e)
        return get_json_result(retcode=-104, retmsg="failed", job_id=_job_id)
=======
@manager.route('/<access_module>', methods=['post'])
def download_upload(access_module):
    request_config = request.json
    required_arguments = ['work_mode', 'namespace', 'table_name']
    if access_module == 'upload':
        required_arguments.extend(['file', 'head', 'partition'])
    elif access_module == 'download':
        required_arguments.extend(['output_path'])
    else:
        raise Exception('can not support this operating: {}'.format(access_module))
    detect_utils.check_config(request_config, required_arguments=required_arguments)

    if access_module == "upload":
        if not os.path.isabs(request_config['file']):
            request_config["file"] = os.path.join(file_utils.get_project_base_directory(), request_config["file"])
    job_dsl, job_runtime_conf = gen_data_access_job_config(request_config, access_module)
    job_id, job_dsl_path, job_runtime_conf_path, model_info, board_url = JobController.submit_job({'job_dsl': job_dsl, 'job_runtime_conf': job_runtime_conf})
    return get_json_result(job_id=job_id, data={'job_dsl_path': job_dsl_path,
                                                'job_runtime_conf_path': job_runtime_conf_path,
                                                'board_url': board_url
                                                })


def gen_data_access_job_config(config_data, access_module):
    job_runtime_conf = {
        "initiator": {},
        "job_parameters": {},
        "role": {},
        "role_parameters": {}
    }
    initiator_role = "local"
    initiator_party_id = 0
    job_runtime_conf["initiator"]["role"] = initiator_role
    job_runtime_conf["initiator"]["party_id"] = initiator_party_id
    job_runtime_conf["job_parameters"]["work_mode"] = config_data["work_mode"]
    job_runtime_conf["role"][initiator_role] = [initiator_party_id]
    job_dsl = {
        "components": {}
    }

    if access_module == 'upload':
        if not os.path.isabs(config_data["file"]):
            config_data["file"] = os.path.join(file_utils.get_project_base_directory(), config_data["file"])
        job_runtime_conf["role_parameters"][initiator_role] = {
            "upload_0": {
                "head": [config_data["head"]],
                "partition": [config_data["partition"]],
                "file": [config_data.get("file")],
                "namespace": [config_data["namespace"]],
                "table_name": [config_data["table_name"]],
            }
        }

        job_dsl["components"]["upload_0"] = {
            "module": "Upload"
        }

    if access_module == 'download':
        job_runtime_conf["role_parameters"][initiator_role] = {
            "download_0": {
                "output_path": [config_data.get("output_path")],
                "namespace": [config_data.get("namespace")],
                "table_name": [config_data.get("table_name")]
            }
        }

        job_dsl["components"]["download_0"] = {
            "module": "Download"
        }

    return job_dsl, job_runtime_conf
>>>>>>> fate_flow: improve data access interface
