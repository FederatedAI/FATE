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
from flask import request

from fate_flow.db.db_models import Task
from fate_flow.operation.job_saver import JobSaver
from fate_flow.scheduler.dag_scheduler import DAGScheduler
from fate_flow.utils.api_utils import get_json_result


# apply initiator for control operation


@manager.route('/<job_id>/<role>/<party_id>/stop/<stop_status>', methods=['POST'])
def stop_job(job_id, role, party_id, stop_status):
    retcode, retmsg = DAGScheduler.stop_job(job_id=job_id, role=role, party_id=party_id, stop_status=stop_status)
    return get_json_result(retcode=retcode, retmsg=retmsg)


@manager.route('/<job_id>/<role>/<party_id>/rerun', methods=['POST'])
def rerun_job(job_id, role, party_id):
    DAGScheduler.set_job_rerun(job_id=job_id, initiator_role=role, initiator_party_id=party_id,
                               component_name=request.json.get("component_name"),
                               force=request.json.get("force", False),
                               auto=False)
    #todo: 判断状态
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
    JobSaver.update_task(task_info=task_info)
    if task_info.get("party_status"):
        JobSaver.update_status(Task, task_info)
    return get_json_result(retcode=0, retmsg='success')
