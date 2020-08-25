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

from fate_arch.common.base_utils import deserialize_b64
from fate_flow.operation.job_tracker import Tracker
from fate_flow.settings import stat_logger
from fate_flow.utils.api_utils import get_json_result

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/metric_data/save',
               methods=['POST'])
def save_metric_data(job_id, component_name, task_version, task_id, role, party_id):
    request_data = request.json
    tracker = Tracker(job_id=job_id, component_name=component_name, task_id=task_id, task_version=task_version,
                      role=role, party_id=party_id)
    metrics = [deserialize_b64(metric) for metric in request_data['metrics']]
    tracker.save_metric_data(metric_namespace=request_data['metric_namespace'], metric_name=request_data['metric_name'],
                             metrics=metrics, job_level=request_data['job_level'])
    return get_json_result()


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/metric_meta/save',
               methods=['POST'])
def save_metric_meta(job_id, component_name, task_version, task_id, role, party_id):
    request_data = request.json
    tracker = Tracker(job_id=job_id, component_name=component_name, task_id=task_id, task_version=task_version,
                      role=role, party_id=party_id)
    metric_meta = deserialize_b64(request_data['metric_meta'])
    tracker.save_metric_meta(metric_namespace=request_data['metric_namespace'], metric_name=request_data['metric_name'],
                             metric_meta=metric_meta, job_level=request_data['job_level'])
    return get_json_result()


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/output_data_info/save',
               methods=['POST'])
def save_output_data_info(job_id, component_name, task_version, task_id, role, party_id):
    request_data = request.json
    tracker = Tracker(job_id=job_id, component_name=component_name, task_id=task_id, task_version=task_version,
                      role=role, party_id=party_id)
    tracker.insert_output_data_info_into_db(data_name=request_data["data_name"],
                                            table_namespace=request_data["table_namespace"],
                                            table_name=request_data["table_name"])
    return get_json_result()


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/output_data_info/read',
               methods=['POST'])
def read_output_data_info(job_id, component_name, task_version, task_id, role, party_id):
    request_data = request.json
    tracker = Tracker(job_id=job_id, component_name=component_name, task_id=task_id, task_version=task_version,
                      role=role, party_id=party_id)
    output_data_infos = tracker.read_output_data_info_from_db(data_name=request_data["data_name"])
    response_data = []
    for output_data_info in output_data_infos:
        response_data.append(output_data_info.to_human_model_dict())
    return get_json_result(data=response_data)


@manager.route('/<job_id>/<component_name>/<task_id>/<task_version>/<role>/<party_id>/summary/save',
               methods=['POST'])
def save_component_summary(job_id: str, component_name: str, task_version: int, task_id: str, role: str, party_id: int):
    request_data = request.json
    tracker = Tracker(job_id=job_id, component_name=component_name, task_id=task_id, task_version=task_version,
                      role=role, party_id=party_id)
    summary_data = request_data['summary']
    tracker.insert_summary_into_db(summary_data)
    return get_json_result()
