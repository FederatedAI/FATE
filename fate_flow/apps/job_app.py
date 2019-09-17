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
import io
import os
import tarfile

from flask import Flask, request, send_file

from arch.api.utils.core import json_loads
from fate_flow.driver.job_controller import JobController
from fate_flow.driver.task_scheduler import TaskScheduler
from fate_flow.settings import stat_logger, CLUSTER_STANDALONE_JOB_SERVER_PORT
from fate_flow.utils import job_utils, detect_utils
from fate_flow.utils.api_utils import get_json_result, request_execute_server
from fate_flow.entity.constant_config import WorkMode
from fate_flow.entity.runtime_config import RuntimeConfig

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/submit', methods=['POST'])
def submit_job():
    work_mode = request.json.get('job_runtime_conf', {}).get('job_parameters', {}).get('work_mode', None)
    detect_utils.check_config({'work_mode': work_mode}, required_arguments=[('work_mode', (WorkMode.CLUSTER, WorkMode.STANDALONE))])
    if work_mode == RuntimeConfig.WORK_MODE:
        job_id, job_dsl_path, job_runtime_conf_path, model_info, board_url = JobController.submit_job(request.json)
        return get_json_result(job_id=job_id, data={'job_dsl_path': job_dsl_path,
                                                    'job_runtime_conf_path': job_runtime_conf_path,
                                                    'model_info': model_info,
                                                    'board_url': board_url
                                                    })
    else:
        if RuntimeConfig.WORK_MODE == WorkMode.CLUSTER and work_mode == WorkMode.STANDALONE:
            # use cluster standalone job server to execute standalone job
            return request_execute_server(request=request, execute_host='{}:{}'.format(request.remote_addr, CLUSTER_STANDALONE_JOB_SERVER_PORT))
        else:
            raise Exception('server run on standalone can not support cluster mode job')


@manager.route('/stop', methods=['POST'])
@job_utils.job_server_routing()
def stop_job():
    TaskScheduler.stop_job(job_id=request.json.get('job_id', ''))
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/query', methods=['POST'])
def query_job():
    jobs = job_utils.query_job(**request.json)
    if not jobs:
        return get_json_result(retcode=101, retmsg='find job failed')
    return get_json_result(retcode=0, retmsg='success', data=[job.to_json() for job in jobs])


@manager.route('/config', methods=['POST'])
def job_config():
    jobs = job_utils.query_job(**request.json)
    if not jobs:
        return get_json_result(retcode=101, retmsg='find job failed')
    else:
        job = jobs[0]
        response_data = dict()
        response_data['job_id'] = job.f_job_id
        response_data['dsl'] = json_loads(job.f_dsl)
        response_data['runtime_conf'] = json_loads(job.f_runtime_conf)
        response_data['train_runtime_conf'] = json_loads(job.f_train_runtime_conf)
        response_data['model_info'] = {'model_id': response_data['runtime_conf']['job_parameters']['model_id'],
                                       'model_version': response_data['runtime_conf']['job_parameters'][
                                           'model_version']}
        return get_json_result(retcode=0, retmsg='success', data=response_data)


@manager.route('/log', methods=['get'])
@job_utils.job_server_routing(307)
def job_log():
    job_id = request.json.get('job_id', '')
    memory_file = io.BytesIO()
    tar = tarfile.open(fileobj=memory_file, mode='w:gz')
    job_log_dir = job_utils.get_job_log_directory(job_id=job_id)
    for root, dir, files in os.walk(job_log_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, job_log_dir)
            tar.add(full_path, rel_path)
    tar.close()
    memory_file.seek(0)
    return send_file(memory_file, attachment_filename='job_{}_log.tar.gz'.format(job_id), as_attachment=True)


@manager.route('/task/query', methods=['POST'])
def query_task():
    tasks = job_utils.query_task(**request.json)
    if not tasks:
        return get_json_result(retcode=101, retmsg='find task failed')
    return get_json_result(retcode=0, retmsg='success', data=[task.to_json() for task in tasks])
