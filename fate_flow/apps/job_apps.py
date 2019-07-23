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
from fate_flow.utils.api_utils import get_json_result
from arch.api.utils.core import base64_decode
from flask import Flask, request
from fate_flow.settings import stat_logger
from fate_flow.driver.job_controller import JobController

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/submit', methods=['POST'])
def submit_job():
    job_id, job_dsl_path, job_runtime_conf_path, model_id, model_version = JobController.submit_job(request.json)
    return get_json_result(job_id=job_id, data={'job_dsl_path': job_dsl_path,
                                                'job_runtime_conf_path': job_runtime_conf_path,
                                                'model_id': model_id,
                                                'model_version': model_version
                                                })


@manager.route('/<job_id>/<role>/<party_id>/create', methods=['POST'])
def create_job(job_id, role, party_id):
    JobController.job_status(job_id=job_id, role=role, party_id=int(party_id), job_info=request.json, create=True)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<role>/<party_id>/status', methods=['POST'])
def job_status(job_id, role, party_id):
    JobController.job_status(job_id=job_id, role=role, party_id=int(party_id), job_info=request.json, create=False)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<role>/<party_id>/<model_id>/save/pipeline', methods=['POST'])
def save_pipeline(job_id, role, party_id, model_id):
    JobController.save_pipeline(job_id=job_id, role=role, party_id=party_id, model_key=base64_decode(model_id))
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<role>/<party_id>/clean', methods=['POST'])
def clean(job_id, role, party_id):
    JobController.clean_job(job_id=job_id, role=role, party_id=party_id)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<component_name>/<task_id>/<role>/<party_id>/run', methods=['POST'])
def run_task(job_id, component_name, task_id, role, party_id):
    task_data = request.json
    task_data['request_url_without_host'] = request.url.lstrip(request.host_url)
    JobController.start_task(job_id, component_name, task_id, role, party_id, request.json)
    return get_json_result(retcode=0, retmsg='success')


@manager.route('/<job_id>/<component_name>/<task_id>/<role>/<party_id>/status', methods=['POST'])
def task_status(job_id, component_name, task_id, role, party_id):
    JobController.task_status(job_id, component_name, task_id, role, party_id, request.json)
    return get_json_result(retcode=0, retmsg='success')
