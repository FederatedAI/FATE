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
import os
import shutil
from arch.task_manager.job_manager import save_job_info, update_job_queue, pop_from_job_queue, \
    get_job_directory, clean_job, set_job_failed, new_runtime_conf, generate_job_id, push_into_job_queue, run_subprocess
from arch.task_manager.utils.api_utils import get_json_result, federated_api
from arch.task_manager.settings import logger, DEFAULT_WORKFLOW_DATA_TYPE
from arch.api.utils import dtable_utils
import copy
from arch.api.utils import file_utils
import json
import datetime
import psutil
from psutil import NoSuchProcess

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    logger.exception(e)
    return get_json_result(status=100, msg=str(e))


@manager.route('/workflow', methods=['POST'])
def submit_workflow_job():
    _data = request.json
    _job_id = generate_job_id()
    logger.info('generated job_id {}, body {}'.format(_job_id, _data))
    push_into_job_queue(job_id=_job_id, config=_data)
    return get_json_result(job_id=_job_id)


@manager.route('/<job_id>/<module>/<role>', methods=['POST'])
def start_workflow(job_id, module, role):
    _config = request.json
    _job_dir = get_job_directory(job_id)
    _party_id = str(_config['local']['party_id'])
    _method = _config['WorkFlowParam']['method']
    default_runtime_dict = file_utils.load_json_conf('workflow/conf/default_runtime_conf.json')
    fill_runtime_conf_table_info(runtime_conf=_config, default_runtime_conf=default_runtime_dict)
    conf_file_path = new_runtime_conf(job_dir=_job_dir, method=_method, module=module, role=role, party_id=_party_id)
    with open(conf_file_path, 'w+') as f:
        f.truncate()
        f.write(json.dumps(_config, indent=4))
        f.flush()
    progs = ["python3",
             os.path.join(file_utils.get_project_base_directory(), _config['CodePath']),
             "-j", job_id,
             "-c", os.path.abspath(conf_file_path)
             ]
    p = run_subprocess(job_dir=_job_dir, job_role=role, progs=progs)
    job_status = "start"
    job_data = dict()
    job_data["begin_date"] = datetime.datetime.now()
    job_data["status"] = job_status
    job_data.update(_config)
    job_data["pid"] = p.pid
    job_data["all_party"] = json.dumps(_config.get("role", {}))
    job_data["initiator"] = _config.get("JobParam", {}).get("initiator")
    save_job_info(job_id=job_id,
                  role=_config.get("local", {}).get("role"),
                  party_id=_config.get("local", {}).get("party_id"),
                  save_info=job_data,
                  create=True)
    update_job_queue(job_id=job_id,
                     role=role,
                     party_id=_party_id,
                     save_data={"status": job_status, "pid": p.pid})
    return get_json_result(data={'pid': p.pid}, job_id=job_id)


@manager.route('/<job_id>/<role>/<party_id>', methods=['DELETE'])
def stop_workflow(job_id, role, party_id):
    _job_dir = get_job_directory(job_id)
    task_pid_path = os.path.join(_job_dir, 'pids')
    if os.path.isdir(task_pid_path):
        for pid_file in os.listdir(task_pid_path):
            try:
                if not pid_file.endswith('.pid'):
                    continue
                with open(os.path.join(task_pid_path, pid_file), 'r') as f:
                    pids = f.read().split('\n')
                    for pid in pids:
                        try:
                            if len(pid) == 0:
                                continue
                            logger.debug("terminating process pid:{} {}".format(pid, pid_file))
                            p = psutil.Process(int(pid))
                            for child in p.children(recursive=True):
                                child.kill()
                            p.kill()
                        except NoSuchProcess:
                            continue
            except Exception as e:
                logger.exception("error")
                continue
        federated_api(job_id=job_id,
                      method='POST',
                      url='/job/jobStatus/{}/{}/{}'.format(job_id, role, party_id),
                      party_id=party_id,
                      json_body={'status': 'failed', 'stopJob': True})
        clean_job(job_id=job_id)
    return get_json_result(job_id=job_id)


def fill_runtime_conf_table_info(runtime_conf, default_runtime_conf):
    if not runtime_conf.get('gen_table_info'):
        return
    table_config = copy.deepcopy(runtime_conf)
    workflow_param = runtime_conf.get('WorkFlowParam')
    default_workflow_param = default_runtime_conf.get('WorkFlowParam')
    for data_type in DEFAULT_WORKFLOW_DATA_TYPE:
        name_param = '{}_table'.format(data_type)
        namespace_param = '{}_namespace'.format(data_type)
        table_config['data_type'] = data_type
        input_output = data_type.split('_')[-1]
        if (not workflow_param.get(name_param)
            or workflow_param.get(name_param) == default_workflow_param.get(name_param)) \
                and (not workflow_param.get(namespace_param)
                     or workflow_param.get(namespace_param) == default_workflow_param.get(namespace_param)):
            if input_output == 'input':
                _create = False
                table_config['table_name'] = ''
            else:
                _create = True
                table_config['table_name'] = runtime_conf.get('JobParam', {}).get('job_id')
            table_name, namespace = dtable_utils.get_table_info(config=table_config, create=_create)
            workflow_param[name_param] = table_name
            workflow_param[namespace_param] = namespace


@manager.route('/workflowRuntimeConf/<job_id>', methods=['POST'])
def get_runtime_conf(job_id):
    _job_dir = get_job_directory(job_id)
    response_data = []
    shutil.copytree(_job_dir, job_id)
    return get_json_result(data=response_data, job_id=job_id)
