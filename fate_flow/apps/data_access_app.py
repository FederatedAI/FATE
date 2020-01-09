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

from arch.api import session
from fate_flow.settings import stat_logger
from fate_flow.utils.api_utils import get_json_result
from fate_flow.utils import detect_utils, job_utils
from fate_flow.driver.job_controller import JobController
from fate_flow.utils.job_utils import get_job_configuration

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


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
    data = {}
    if access_module == "upload":
        data['table_name'] = request_config["table_name"]
        data['namespace'] = request_config["namespace"]
    job_dsl, job_runtime_conf = gen_data_access_job_config(request_config, access_module)
    job_id, job_dsl_path, job_runtime_conf_path, logs_directory, model_info, board_url = JobController.submit_job({'job_dsl': job_dsl, 'job_runtime_conf': job_runtime_conf})
    data.update({'job_dsl_path': job_dsl_path, 'job_runtime_conf_path': job_runtime_conf_path,
                 'board_url': board_url, 'logs_directory': logs_directory})
    return get_json_result(job_id=job_id, data=data)


@manager.route('/upload/history', methods=['POST'])
def upload_history():
    data = get_upload_history()
    return get_json_result(retcode=0, retmsg='success', data=data)


def get_upload_history():
    request_data = request.json
    if request_data.get('job_id'):
        tasks = job_utils.query_task(component_name='upload_0', status='success', job_id=request_data.get('job_id'))
    else:
        tasks = job_utils.query_task(component_name='upload_0', status='success')
    limit= request_data.get('limit')
    if not limit:
        tasks = tasks[-1::-1]
    else:
        tasks = tasks[-1:-limit - 1:-1]
    jobs_run_conf = get_job_configuration(None, None, None, tasks)
    return get_upload_info(jobs_run_conf)


def get_upload_info(jobs_run_conf):
    data = []
    for job_id, job_run_conf in jobs_run_conf.items():
        info = {}
        table_name = job_run_conf["table_name"][0]
        namespace = job_run_conf["namespace"][0]
        partition = job_run_conf["partition"][0]
        info["upload_info"] = {
            "table_name": table_name,
            "namespace": namespace,
            "partition": partition,
        }
        info["notes"] = job_run_conf["notes"]
        info["meta"] = session.get_data_table_metas(table_name, namespace)
        data.append({job_id: info})
    return data


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
        job_runtime_conf["role_parameters"][initiator_role] = {
            "upload_0": {
                "work_mode": [config_data["work_mode"]],
                "head": [config_data["head"]],
                "partition": [config_data["partition"]],
                "file": [config_data["file"]],
                "namespace": [config_data["namespace"]],
                "table_name": [config_data["table_name"]],
                "in_version": [config_data.get("in_version")],
            }
        }

        job_dsl["components"]["upload_0"] = {
            "module": "Upload"
        }

    if access_module == 'download':
        job_runtime_conf["role_parameters"][initiator_role] = {
            "download_0": {
                "work_mode": [config_data["work_mode"]],
                "delimitor": [config_data.get("delimitor", ",")],
                "output_path": [config_data["output_path"]],
                "namespace": [config_data["namespace"]],
                "table_name": [config_data["table_name"]]
            }
        }

        job_dsl["components"]["download_0"] = {
            "module": "Download"
        }

    return job_dsl, job_runtime_conf
