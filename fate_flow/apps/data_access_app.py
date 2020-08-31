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

from flask import Flask, request

from fate_arch import storage
from fate_arch.storage import StorageEngine
from fate_flow.entity.types import StatusSet
from fate_flow.operation import JobSaver
from fate_flow.scheduler import DAGScheduler
from fate_flow.settings import stat_logger, USE_LOCAL_DATA, WORK_MODE
from fate_flow.utils import detect_utils, job_utils
from fate_flow.utils.api_utils import get_json_result

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/<access_module>', methods=['post'])
def download_upload(access_module):
    job_id = job_utils.generate_job_id()
    if access_module == "upload" and USE_LOCAL_DATA and not (request.json and request.json.get("use_local_data") == 0):
        file = request.files['file']
        filename = os.path.join(job_utils.get_job_directory(job_id), 'fate_upload_tmp', file.filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        try:
            file.save(filename)
        except Exception as e:
            shutil.rmtree(os.path.join(job_utils.get_job_directory(job_id), 'tmp'))
            raise e
        job_config = request.args.to_dict()
        job_config['file'] = filename
    else:
        job_config = request.json
    required_arguments = ['work_mode', 'namespace', 'table_name']
    if access_module == 'upload':
        required_arguments.extend(['file', 'head', 'partition'])
    elif access_module == 'download':
        required_arguments.extend(['output_path'])
    else:
        raise Exception('can not support this operating: {}'.format(access_module))
    detect_utils.check_config(job_config, required_arguments=required_arguments)
    data = {}
    for _ in ["work_mode", "backend", "drop"]:
        if _ in job_config:
            job_config[_] = int(job_config[_])
    if access_module == "upload":
        data['table_name'] = job_config["table_name"]
        data['namespace'] = job_config["namespace"]
        if WORK_MODE != 0:
            job_config["storage_engine"] = job_config.get("storage_engine", StorageEngine.EGGROLL)
            data_table_meta = storage.StorageTableMeta(name=job_config["table_name"], namespace=job_config["namespace"])
            if data_table_meta and job_config.get('drop', 2) == 2:
                return get_json_result(retcode=100,
                                       retmsg='The data table already exists.'
                                              'If you still want to continue uploading, please add the parameter -drop.'
                                              ' 0 means not to delete and continue uploading, '
                                              '1 means to upload again after deleting the table')
            elif data_table_meta and job_config.get('drop', 2) == 1:
                job_config["destroy"] = True
        else:
            job_config["storage_engine"] = job_config.get("storage_engine", StorageEngine.STANDALONE)
    # compatibility
    if "table_name" in job_config:
        job_config["name"] = job_config["table_name"]
    job_dsl, job_runtime_conf = gen_data_access_job_config(job_config, access_module)
    job_id, job_dsl_path, job_runtime_conf_path, logs_directory, model_info, board_url = DAGScheduler.submit(
        {'job_dsl': job_dsl, 'job_runtime_conf': job_runtime_conf}, job_id=job_id)
    data.update({'job_dsl_path': job_dsl_path, 'job_runtime_conf_path': job_runtime_conf_path,
                 'board_url': board_url, 'logs_directory': logs_directory})
    return get_json_result(job_id=job_id, data=data)


@manager.route('/upload/history', methods=['POST'])
def upload_history():
    request_data = request.json
    if request_data.get('job_id'):
        tasks = JobSaver.query_task(component_name='upload_0', status=StatusSet.COMPLETE, job_id=request_data.get('job_id'), run_on=True)
    else:
        tasks = JobSaver.query_task(component_name='upload_0', status=StatusSet.COMPLETE, run_on=True)
    limit = request_data.get('limit')
    if not limit:
        tasks = tasks[-1::-1]
    else:
        tasks = tasks[-1:-limit - 1:-1]
    jobs_run_conf = job_utils.get_job_configuration(None, None, None, tasks)
    data = get_upload_info(jobs_run_conf=jobs_run_conf)
    return get_json_result(retcode=0, retmsg='success', data=data)


def get_upload_info(jobs_run_conf):
    data = []

    for job_id, job_run_conf in jobs_run_conf.items():
        info = {}
        table_name = job_run_conf["name"][0]
        namespace = job_run_conf["namespace"][0]
        table_meta = storage.StorageTableMeta(name=table_name, namespace=namespace)
        if table_meta:
            partition = job_run_conf["partition"][0]
            info["upload_info"] = {
                "table_name": table_name,
                "namespace": namespace,
                "partition": partition,
                'upload_count': table_meta.get_count()
            }
            info["notes"] = job_run_conf["notes"]
            info["schema"] = table_meta.get_schema()
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
    initiator_party_id = config_data.get('party_id', 0)
    job_runtime_conf["initiator"]["role"] = initiator_role
    job_runtime_conf["initiator"]["party_id"] = initiator_party_id
    for _ in ["work_mode", "backend"]:
        if _ in config_data:
            job_runtime_conf["job_parameters"][_] = config_data[_]
    job_runtime_conf["role"][initiator_role] = [initiator_party_id]
    job_dsl = {
        "components": {}
    }

    if access_module == 'upload':
        parameters = {
            "upload_0": {
                "work_mode": [int(config_data["work_mode"])],
                "head": [int(config_data["head"])],
                "partition": [int(config_data["partition"])],
                "file": [config_data["file"]],
                "namespace": [config_data["namespace"]],
                "name": [config_data["name"]],
                "storage_engine": [config_data.get("storage_engine", StorageEngine.EGGROLL)],
                "storage_address": [config_data.get("storage_address", None)],
                "destroy": [config_data.get("destroy", False)],
            }
        }
        if int(config_data.get('dsl_version', 1)) == 2:
            job_runtime_conf['algorithm_parameters'] = parameters
            job_runtime_conf['job_parameters']['dsl_version'] = 2
        else:
            job_runtime_conf["role_parameters"][initiator_role] = parameters
            job_runtime_conf['job_parameters']['dsl_version'] = 1
        job_dsl["components"]["upload_0"] = {
            "module": "Upload"
        }

    if access_module == 'download':
        parameters = {
            "download_0": {
                "work_mode": [config_data["work_mode"]],
                "delimitor": [config_data.get("delimitor", ",")],
                "output_path": [config_data["output_path"]],
                "namespace": [config_data["namespace"]],
                "name": [config_data["name"]]
            }
        }
        if int(config_data.get('dsl_version', 1)) == 2:
            job_runtime_conf['algorithm_parameters'] = parameters
            job_runtime_conf['job_parameters']['dsl_version'] = 2
        else:
            job_runtime_conf["role_parameters"][initiator_role] = parameters
            job_runtime_conf['job_parameters']['dsl_version'] = 1
        job_dsl["components"]["download_0"] = {
            "module": "Download"
        }

    return job_dsl, job_runtime_conf
