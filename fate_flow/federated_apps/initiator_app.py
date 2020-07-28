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

from arch.api.utils.core_utils import json_loads, json_dumps
from fate_flow.scheduler.task_scheduler import TaskScheduler
from fate_flow.scheduler.dag_scheduler import DAGScheduler
from fate_flow.scheduler.federated_scheduler import FederatedScheduler
from fate_flow.controller.task_controller import TaskController
from fate_flow.operation.job_saver import JobSaver
from fate_flow.manager import data_manager
from fate_flow.settings import stat_logger, CLUSTER_STANDALONE_JOB_SERVER_PORT
from fate_flow.utils import job_utils, detect_utils
from fate_flow.utils.api_utils import get_json_result, request_execute_server
from fate_flow.entity.constant import WorkMode, JobStatus, RetCode, FederatedSchedulingStatusCode
from fate_flow.entity.runtime_config import RuntimeConfig

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))

# API for apply do something


@manager.route('/<job_id>/<role>/<party_id>/stop/<stop_status>', methods=['POST'])
def stop_job(job_id, role, party_id, stop_status):
    jobs = job_utils.query_job(job_id=job_id, role=role, party_id=party_id, is_initiator=1)
    if len(jobs) > 0:
        status_code, response = FederatedScheduler.stop_job(job=jobs[0], stop_status=stop_status)
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            return get_json_result(retcode=0, retmsg='success')
        else:
            return get_json_result(retcode=RetCode.FEDERATED_ERROR, retmsg=json_dumps(response))
    else:
        return get_json_result(retcode=RetCode.OPERATING_ERROR, retmsg="can not found job")


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/status', methods=['POST'])
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
    JobSaver.update_task(task_info=task_info)
    return get_json_result(retcode=0, retmsg='success')
