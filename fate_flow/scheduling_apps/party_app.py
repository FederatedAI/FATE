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

from fate_flow.entity.constant import RetCode
from fate_flow.controller import JobController
from fate_flow.controller import TaskController
from fate_flow.settings import stat_logger
from fate_flow.utils.api_utils import get_json_result
from fate_flow.utils.authentication_utils import request_authority_certification
from fate_flow.operation import JobSaver
from fate_arch.common import log

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=RetCode.EXCEPTION_ERROR, retmsg=log.exception_to_trace_string(e))


# execute command on every party
@manager.route('/<job_id>/<role>/<party_id>/create', methods=['POST'])
@request_authority_certification
def create_job(job_id, role, party_id):
    JobController.create_job(job_id=job_id, role=role, party_id=int(party_id), job_info=request.json)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<role>/<party_id>/resource/apply', methods=['POST'])
def apply_resource(job_id, role, party_id):
    status = JobController.apply_resource(job_id=job_id, role=role, party_id=int(party_id))
    if status:
        return get_json_result(retcode=0, retmsg='success')
    else:
        return get_json_result(retcode=RetCode.OPERATING_ERROR, retmsg=f"apply for job {job_id} resource failed")


@manager.route('/<job_id>/<role>/<party_id>/resource/return', methods=['POST'])
def return_resource(job_id, role, party_id):
    status = JobController.return_resource(job_id=job_id, role=role, party_id=int(party_id))
    if status:
        return get_json_result(retcode=0, retmsg='success')
    else:
        return get_json_result(retcode=RetCode.OPERATING_ERROR, retmsg=f"apply for job {job_id} resource failed")


@manager.route('/<job_id>/<role>/<party_id>/start', methods=['POST'])
def start_job(job_id, role, party_id):
    JobController.start_job(job_id=job_id, role=role, party_id=int(party_id))
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<role>/<party_id>/update', methods=['POST'])
def update_job(job_id, role, party_id):
    job_info = {}
    job_info.update(request.json)
    job_info.update({
        "job_id": job_id,
        "role": role,
        "party_id": party_id
    })
    JobController.update_job(job_info=job_info)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<role>/<party_id>/status/<status>', methods=['POST'])
def job_status(job_id, role, party_id, status):
    job_info = {}
    job_info.update({
        "job_id": job_id,
        "role": role,
        "party_id": party_id,
        "status": status
    })
    if JobController.update_job_status(job_info=job_info):
        return get_json_result(retcode=0, retmsg='success')
    else:
        return get_json_result(retcode=RetCode.OPERATING_ERROR, retmsg="update job status failed")


@manager.route('/<job_id>/<role>/<party_id>/model', methods=['POST'])
@request_authority_certification
def save_pipelined_model(job_id, role, party_id):
    JobController.save_pipelined_model(job_id=job_id, role=role, party_id=party_id)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<role>/<party_id>/stop/<stop_status>', methods=['POST'])
def stop_job(job_id, role, party_id, stop_status):
    jobs = JobSaver.query_job(job_id=job_id, role=role, party_id=party_id)
    for job in jobs:
        JobController.stop_job(job=job, stop_status=stop_status)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<role>/<party_id>/cancel', methods=['POST'])
def cancel_job(job_id, role, party_id):
    status = JobController.cancel_job(job_id=job_id, role=role, party_id=int(party_id))
    if status:
        return get_json_result(retcode=0, retmsg='cancel job success')
    else:
        return get_json_result(retcode=RetCode.OPERATING_ERROR, retmsg='cancel job failed')


@manager.route('/<job_id>/<role>/<party_id>/clean', methods=['POST'])
@request_authority_certification
def clean(job_id, role, party_id):
    JobController.clean_job(job_id=job_id, role=role, party_id=party_id, roles=request.json)
    return get_json_result(retcode=0, retmsg='success')


# Control API for task
@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/create', methods=['POST'])
@request_authority_certification
def create_task(job_id, component_name, task_id, task_version, role, party_id):
    TaskController.create_task(role, party_id, True, request.json)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/start', methods=['POST'])
@request_authority_certification
def start_task(job_id, component_name, task_id, task_version, role, party_id):
    TaskController.start_task(job_id, component_name, task_id, task_version, role, party_id, request.json)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/report', methods=['POST'])
def report_task(job_id, component_name, task_id, task_version, role, party_id):
    task_info = {}
    task_info.update(request.json)
    task_info.update({
        "job_id": job_id,
        "task_id": task_id,
        "task_version": task_version,
        "role": role,
        "party_id": party_id,
    })
    TaskController.update_task(task_info=task_info)
    if task_info.get("party_status"):
        if not TaskController.update_task_status(task_info=task_info):
            return get_json_result(retcode=RetCode.OPERATING_ERROR, retmsg="update task status failed")
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/update', methods=['POST'])
def update_task(job_id, component_name, task_id, task_version, role, party_id):
    task_info = {}
    task_info.update(request.json)
    task_info.update({
        "job_id": job_id,
        "task_id": task_id,
        "task_version": task_version,
        "role": role,
        "party_id": party_id,
    })
    TaskController.update_task(task_info=task_info)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/status/<status>', methods=['POST'])
def task_status(job_id, component_name, task_id, task_version, role, party_id, status):
    task_info = {}
    task_info.update({
        "job_id": job_id,
        "task_id": task_id,
        "task_version": task_version,
        "role": role,
        "party_id": party_id,
        "status": status
    })
    if TaskController.update_task_status(task_info=task_info):
        return get_json_result(retcode=0, retmsg='success')
    else:
        return get_json_result(retcode=RetCode.OPERATING_ERROR, retmsg="update task status failed")


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/stop/<stop_status>', methods=['POST'])
def stop_task(job_id, component_name, task_id, task_version, role, party_id, stop_status):
    tasks = JobSaver.query_task(job_id=job_id, task_id=task_id, task_version=task_version, role=role, party_id=int(party_id))
    for task in tasks:
        TaskController.stop_task(task=task, stop_status=stop_status)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/clean/<content_type>', methods=['POST'])
def clean_task(job_id, component_name, task_id, task_version, role, party_id, content_type):
    tasks = JobSaver.query_task(job_id=job_id, task_id=task_id, task_version=task_version, role=role, party_id=int(party_id))
    for task in tasks:
        TaskController.clean_task(task=task, content_type=content_type)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/input/args', methods=['POST'])
def query_task_input_args(job_id, component_name, task_id, task_version, role, party_id):
    task_input_args = TaskController.query_task_input_args(job_id, task_id, role, party_id,
                                                          job_args=request.json.get('job_args', {}),
                                                          job_parameters=request.json.get('job_parameters', {}),
                                                          input_dsl=request.json.get('input', {}),
                                                          filter_type=['data'],
                                                          filter_attr={'data': ['partitions']})
    return get_json_result(retcode=0, retmsg='success', data=task_input_args)
