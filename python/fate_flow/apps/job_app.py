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
import json
import tarfile

from flask import Flask, request, send_file

from fate_arch.common import WorkMode
from fate_arch.common.base_utils import json_loads, json_dumps
from fate_flow.scheduler import DAGScheduler
from fate_flow.scheduler import FederatedScheduler
from fate_flow.settings import stat_logger, TEMP_DIRECTORY
from fate_flow.utils import job_utils, detect_utils, schedule_utils
from fate_flow.utils.api_utils import get_json_result, error_response
from fate_flow.entity.types import FederatedSchedulingStatusCode, RetCode, JobStatus
from fate_flow.operation import Tracker
from fate_flow.operation import JobSaver
from fate_flow.operation import JobClean

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/submit', methods=['POST'])
def submit_job():
    work_mode = request.json.get('job_runtime_conf', {}).get('job_parameters', {}).get('work_mode', None)
    detect_utils.check_config({'work_mode': work_mode}, required_arguments=[('work_mode', (WorkMode.CLUSTER, WorkMode.STANDALONE))])
    job_id, job_dsl_path, job_runtime_conf_path, logs_directory, model_info, board_url = DAGScheduler.submit(request.json)
    return get_json_result(retcode=0, retmsg='success',
                           job_id=job_id,
                           data={'job_dsl_path': job_dsl_path,
                                 'job_runtime_conf_path': job_runtime_conf_path,
                                 'model_info': model_info,
                                 'board_url': board_url,
                                 'logs_directory': logs_directory
                                 })


@manager.route('/stop', methods=['POST'])
def stop_job():
    job_id = request.json.get('job_id')
    stop_status = request.json.get("stop_status", "canceled")
    jobs = JobSaver.query_job(job_id=job_id)
    if jobs:
        status_code, response = FederatedScheduler.request_stop_job(job=jobs[0], stop_status=stop_status)
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            return get_json_result(retcode=RetCode.SUCCESS, retmsg="stop job success")
        else:
            return get_json_result(retcode=RetCode.OPERATING_ERROR, retmsg="stop job failed:\n{}".format(json_dumps(response)))
    else:
        return get_json_result(retcode=RetCode.DATA_ERROR, retmsg="can not found job")


@manager.route('/rerun', methods=['POST'])
def rerun_job():
    job_id = request.json.get("job_id")
    jobs = JobSaver.query_job(job_id=job_id)
    if jobs:
        status_code, response = FederatedScheduler.request_rerun_job(job=jobs[0], command_body=request.json)
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            return get_json_result(retcode=RetCode.SUCCESS, retmsg="rerun job success")
        else:
            return get_json_result(retcode=RetCode.OPERATING_ERROR, retmsg="rerun job failed:\n{}".format(json_dumps(response)))
    else:
        return get_json_result(retcode=RetCode.DATA_ERROR, retmsg="can not found job")


@manager.route('/query', methods=['POST'])
def query_job():
    jobs = JobSaver.query_job(**request.json)
    if not jobs:
        return get_json_result(retcode=0, retmsg='no job could be found', data=[])
    return get_json_result(retcode=0, retmsg='success', data=[job.to_json() for job in jobs])


@manager.route('/list/job', methods=['POST'])
def list_job():
    jobs = job_utils.list_job(request.json.get('limit'))
    if not jobs:
        return get_json_result(retcode=101, retmsg='No job found')
    return get_json_result(retcode=0, retmsg='success', data=[job.to_json() for job in jobs])


@manager.route('/update', methods=['POST'])
def update_job():
    job_info = request.json
    jobs = JobSaver.query_job(job_id=job_info['job_id'], party_id=job_info['party_id'], role=job_info['role'])
    if not jobs:
        return get_json_result(retcode=101, retmsg='find job failed')
    else:
        JobSaver.update_job(job_info={'description': job_info.get('notes', ''), 'job_id': job_info['job_id'], 'role': job_info['role'],
                                      'party_id': job_info['party_id']})
        return get_json_result(retcode=0, retmsg='success')


@manager.route('/config', methods=['POST'])
def job_config():
    jobs = JobSaver.query_job(**request.json)
    if not jobs:
        return get_json_result(retcode=101, retmsg='find job failed')
    else:
        job = jobs[0]
        response_data = dict()
        response_data['job_id'] = job.f_job_id
        response_data['dsl'] = job.f_dsl
        response_data['runtime_conf'] = job.f_runtime_conf
        response_data['train_runtime_conf'] = job.f_train_runtime_conf
        response_data['model_info'] = {'model_id': response_data['runtime_conf']['job_parameters']['model_id'],
                                       'model_version': response_data['runtime_conf']['job_parameters'][
                                           'model_version']}
        return get_json_result(retcode=0, retmsg='success', data=response_data)


@manager.route('/log', methods=['get'])
def job_log():
    job_id = request.json.get('job_id', '')
    job_log_dir = job_utils.get_job_log_directory(job_id=job_id)
    if os.path.exists(job_log_dir):
        memory_file = io.BytesIO()
        tar = tarfile.open(fileobj=memory_file, mode='w:gz')
        for root, dir, files in os.walk(job_log_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, job_log_dir)
                tar.add(full_path, rel_path)
        tar.close()
        memory_file.seek(0)
        return send_file(memory_file, attachment_filename='job_{}_log.tar.gz'.format(job_id), as_attachment=True)
    else:
        return error_response(210, "Log file path: {} not found. Please check if the job id is valid.".format(job_log_dir))


@manager.route('/task/query', methods=['POST'])
def query_task():
    tasks = JobSaver.query_task(**request.json)
    if not tasks:
        return get_json_result(retcode=101, retmsg='find task failed')
    return get_json_result(retcode=0, retmsg='success', data=[task.to_json() for task in tasks])


@manager.route('/list/task', methods=['POST'])
def list_task():
    tasks = job_utils.list_task(request.json.get('limit'))
    if not tasks:
        return get_json_result(retcode=100, retmsg='No task found')
    return get_json_result(retcode=0, retmsg='success', data=[task.to_json() for task in tasks])


@manager.route('/data/view/query', methods=['POST'])
def query_component_output_data_info():
    output_data_infos = Tracker.query_output_data_infos(**request.json)
    if not output_data_infos:
        return get_json_result(retcode=101, retmsg='find data view failed')
    return get_json_result(retcode=0, retmsg='success', data=[output_data_info.to_json() for output_data_info in output_data_infos])


@manager.route('/clean', methods=['POST'])
def clean_job():
    JobClean.start_clean_job(**request.json)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/clean/queue', methods=['POST'])
def clean_queue():
    jobs = JobSaver.query_job(is_initiator=True, status=JobStatus.WAITING)
    clean_status = {}
    for job in jobs:
        status_code, response = FederatedScheduler.request_stop_job(job=job, stop_status=JobStatus.CANCELED)
        clean_status[job.f_job_id] = status_code
    return get_json_result(retcode=0, retmsg='success', data=clean_status)


@manager.route('/dsl/generate', methods=['POST'])
def dsl_generator():
    data = request.json
    cpn_str = data.get("cpn_str", "")
    try:
        if not cpn_str:
            raise Exception("Component list should not be empty.")
        if isinstance(cpn_str, list):
            cpn_list = cpn_str
        else:
            if (cpn_str.find("/") and cpn_str.find("\\")) != -1:
                raise Exception("Component list string should not contain '/' or '\\'.")
            cpn_str = cpn_str.replace(" ", "").replace("\n", "").strip(",[]")
            cpn_list = cpn_str.split(",")
        train_dsl = json_loads(data.get("train_dsl"))
        parser = schedule_utils.get_dsl_parser_by_version(data.get("version", "1"))
        predict_dsl = parser.deploy_component(cpn_list, train_dsl)

        if data.get("filename"):
            os.makedirs(TEMP_DIRECTORY, exist_ok=True)
            temp_filepath = os.path.join(TEMP_DIRECTORY, data.get("filename"))
            with open(temp_filepath, "w") as fout:
                fout.write(json.dumps(predict_dsl, indent=4))
            return send_file(open(temp_filepath, 'rb'), as_attachment=True, attachment_filename=data.get("filename"))
        return get_json_result(data=predict_dsl)
    except Exception as e:
        stat_logger.exception(e)
        return error_response(210, "DSL generating failed. For more details, "
                                   "please check logs/fate_flow/fate_flow_stat.log.")


