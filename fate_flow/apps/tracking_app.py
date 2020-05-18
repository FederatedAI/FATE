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
import json
import os
import shutil
import tarfile

from flask import Flask, request, send_file
from google.protobuf import json_format

from arch.api.utils.core_utils import deserialize_b64
from arch.api.utils.core_utils import fate_uuid
from arch.api.utils.core_utils import json_loads
from fate_flow.db.db_models import Job, DB
from fate_flow.manager.data_manager import query_data_view, delete_metric_data
from fate_flow.manager.tracking_manager import Tracking
from fate_flow.settings import stat_logger
from fate_flow.utils import job_utils, data_utils
from fate_flow.utils.api_utils import get_json_result, error_response
from federatedml.feature.instance import Instance

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/job/data_view', methods=['post'])
def job_view():
    request_data = request.json
    check_request_parameters(request_data)
    job_tracker = Tracking(job_id=request_data['job_id'], role=request_data['role'], party_id=request_data['party_id'])
    job_view_data = job_tracker.get_job_view()
    if job_view_data:
        job_metric_list = job_tracker.get_metric_list(job_level=True)
        job_view_data['model_summary'] = {}
        for metric_namespace, namespace_metrics in job_metric_list.items():
            job_view_data['model_summary'][metric_namespace] = job_view_data['model_summary'].get(metric_namespace, {})
            for metric_name in namespace_metrics:
                job_view_data['model_summary'][metric_namespace][metric_name] = job_view_data['model_summary'][
                    metric_namespace].get(metric_name, {})
                for metric_data in job_tracker.get_job_metric_data(metric_namespace=metric_namespace,
                                                                   metric_name=metric_name):
                    job_view_data['model_summary'][metric_namespace][metric_name][metric_data.key] = metric_data.value
        return get_json_result(retcode=0, retmsg='success', data=job_view_data)
    else:
        return get_json_result(retcode=101, retmsg='error')


@manager.route('/component/metric/all', methods=['post'])
def component_metric_all():
    request_data = request.json
    check_request_parameters(request_data)
    tracker = Tracking(job_id=request_data['job_id'], component_name=request_data['component_name'],
                       role=request_data['role'], party_id=request_data['party_id'])
    metrics = tracker.get_metric_list()
    all_metric_data = {}
    if metrics:
        for metric_namespace, metric_names in metrics.items():
            all_metric_data[metric_namespace] = all_metric_data.get(metric_namespace, {})
            for metric_name in metric_names:
                all_metric_data[metric_namespace][metric_name] = all_metric_data[metric_namespace].get(metric_name, {})
                metric_data, metric_meta = get_metric_all_data(tracker=tracker, metric_namespace=metric_namespace,
                                                               metric_name=metric_name)
                all_metric_data[metric_namespace][metric_name]['data'] = metric_data
                all_metric_data[metric_namespace][metric_name]['meta'] = metric_meta
        return get_json_result(retcode=0, retmsg='success', data=all_metric_data)
    else:
        return get_json_result(retcode=0, retmsg='no data', data={})


@manager.route('/component/metrics', methods=['post'])
def component_metrics():
    request_data = request.json
    check_request_parameters(request_data)
    tracker = Tracking(job_id=request_data['job_id'], component_name=request_data['component_name'],
                       role=request_data['role'], party_id=request_data['party_id'])
    metrics = tracker.get_metric_list()
    if metrics:
        return get_json_result(retcode=0, retmsg='success', data=metrics)
    else:
        return get_json_result(retcode=0, retmsg='no data', data={})


@manager.route('/component/metric_data', methods=['post'])
def component_metric_data():
    request_data = request.json
    check_request_parameters(request_data)
    tracker = Tracking(job_id=request_data['job_id'], component_name=request_data['component_name'],
                       role=request_data['role'], party_id=request_data['party_id'])
    metric_data, metric_meta = get_metric_all_data(tracker=tracker, metric_namespace=request_data['metric_namespace'],
                                                   metric_name=request_data['metric_name'])
    if metric_data or metric_meta:
        return get_json_result(retcode=0, retmsg='success', data=metric_data,
                               meta=metric_meta)
    else:
        return get_json_result(retcode=0, retmsg='no data', data=[], meta={})


def get_metric_all_data(tracker, metric_namespace, metric_name):
    metric_data = tracker.get_metric_data(metric_namespace=metric_namespace,
                                          metric_name=metric_name)
    metric_meta = tracker.get_metric_meta(metric_namespace=metric_namespace,
                                          metric_name=metric_name)
    if metric_data or metric_meta:
        metric_data_list = [(metric.key, metric.value) for metric in metric_data]
        metric_data_list.sort(key=lambda x: x[0])
        return metric_data_list, metric_meta.to_dict() if metric_meta else {}
    else:
        return [], {}


@manager.route('/component/metric/delete', methods=['post'])
def component_metric_delete():
    sql = delete_metric_data(request.json)
    return get_json_result(retcode=0, retmsg='success', data=sql)


@manager.route('/component/parameters', methods=['post'])
def component_parameters():
    request_data = request.json
    check_request_parameters(request_data)
    job_id = request_data.get('job_id', '')
    job_dsl_parser = job_utils.get_job_dsl_parser_by_job_id(job_id=job_id)
    if job_dsl_parser:
        component = job_dsl_parser.get_component_info(request_data['component_name'])
        parameters = component.get_role_parameters()
        for role, partys_parameters in parameters.items():
            for party_parameters in partys_parameters:
                if party_parameters.get('local', {}).get('role', '') == request_data['role'] and party_parameters.get(
                        'local', {}).get('party_id', '') == int(request_data['party_id']):
                    output_parameters = {}
                    output_parameters['module'] = party_parameters.get('module', '')
                    for p_k, p_v in party_parameters.items():
                        if p_k.endswith('Param'):
                            output_parameters[p_k] = p_v
                    return get_json_result(retcode=0, retmsg='success', data=output_parameters)
        else:
            return get_json_result(retcode=0, retmsg='can not found this component parameters')
    else:
        return get_json_result(retcode=101, retmsg='can not found this job')


@manager.route('/component/output/model', methods=['post'])
@job_utils.job_server_routing()
def component_output_model():
    request_data = request.json
    check_request_parameters(request_data)
    job_dsl, job_runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=request_data['job_id'],
                                                                                    role=request_data['role'],
                                                                                    party_id=request_data['party_id'])
    model_id = job_runtime_conf['job_parameters']['model_id']
    model_version = job_runtime_conf['job_parameters']['model_version']
    tracker = Tracking(job_id=request_data['job_id'], component_name=request_data['component_name'],
                       role=request_data['role'], party_id=request_data['party_id'], model_id=model_id,
                       model_version=model_version)
    dag = job_utils.get_job_dsl_parser(dsl=job_dsl, runtime_conf=job_runtime_conf,
                                       train_runtime_conf=train_runtime_conf)
    component = dag.get_component_info(request_data['component_name'])
    output_model_json = {}
    # There is only one model output at the current dsl version.
    output_model = tracker.get_output_model(component.get_output()['model'][0] if component.get_output().get('model') else 'default')
    for buffer_name, buffer_object in output_model.items():
        if buffer_name.endswith('Param'):
            output_model_json = json_format.MessageToDict(buffer_object, including_default_value_fields=True)
    if output_model_json:
        component_define = tracker.get_component_define()
        this_component_model_meta = {}
        for buffer_name, buffer_object in output_model.items():
            if buffer_name.endswith('Meta'):
                this_component_model_meta['meta_data'] = json_format.MessageToDict(buffer_object,
                                                                                   including_default_value_fields=True)
        this_component_model_meta.update(component_define)
        return get_json_result(retcode=0, retmsg='success', data=output_model_json, meta=this_component_model_meta)
    else:
        return get_json_result(retcode=0, retmsg='no data', data={})


@manager.route('/component/output/data', methods=['post'])
@job_utils.job_server_routing()
def component_output_data():
    request_data = request.json
    output_data_table = get_component_output_data_table(task_data=request_data)
    if not output_data_table:
        return get_json_result(retcode=0, retmsg='no data', data=[])
    output_data = []
    num = 100
    have_data_label = False
    if output_data_table:
        for k, v in output_data_table.collect():
            if num == 0:
                break
            data_line, have_data_label = get_component_output_data_line(src_key=k, src_value=v)
            output_data.append(data_line)
            num -= 1
    if output_data:
        header = get_component_output_data_meta(output_data_table=output_data_table, have_data_label=have_data_label)
        return get_json_result(retcode=0, retmsg='success', data=output_data, meta={'header': header})
    else:
        return get_json_result(retcode=0, retmsg='no data', data=[])


@manager.route('/component/output/data/download', methods=['get'])
@job_utils.job_server_routing(307)
def component_output_data_download():
    request_data = request.json
    output_data_table = get_component_output_data_table(task_data=request_data)
    limit = request_data.get('limit', -1)
    if not output_data_table:
        return error_response(response_code=500, retmsg='no data')
    if limit == 0:
        return error_response(response_code=500, retmsg='limit is 0')
    output_data_count = 0
    have_data_label = False
    output_tmp_dir = os.path.join(os.getcwd(), 'tmp/{}'.format(fate_uuid()))
    output_file_path = '{}/output_%s'.format(output_tmp_dir)
    output_data_file_path = output_file_path % 'data.csv'
    os.makedirs(os.path.dirname(output_data_file_path), exist_ok=True)
    with open(output_data_file_path, 'w') as fw:
        for k, v in output_data_table.collect():
            data_line, have_data_label = get_component_output_data_line(src_key=k, src_value=v)
            fw.write('{}\n'.format(','.join(map(lambda x: str(x), data_line))))
            output_data_count += 1
            if output_data_count == limit:
                break

    if output_data_count:
        # get meta
        header = get_component_output_data_meta(output_data_table=output_data_table, have_data_label=have_data_label)
        output_data_meta_file_path = output_file_path % 'data_meta.json'
        with open(output_data_meta_file_path, 'w') as fw:
            json.dump({'header': header}, fw, indent=4)
        if request_data.get('head', True):
            with open(output_data_file_path, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write('{}\n'.format(','.join(header)) + content)
        # tar
        memory_file = io.BytesIO()
        tar = tarfile.open(fileobj=memory_file, mode='w:gz')
        tar.add(output_data_file_path, os.path.relpath(output_data_file_path, output_tmp_dir))
        tar.add(output_data_meta_file_path, os.path.relpath(output_data_meta_file_path, output_tmp_dir))
        tar.close()
        memory_file.seek(0)
        try:
            shutil.rmtree(os.path.dirname(output_data_file_path))
        except Exception as e:
            # warning
            stat_logger.warning(e)
        tar_file_name = 'job_{}_{}_{}_{}_output_data.tar.gz'.format(request_data['job_id'],
                                                                    request_data['component_name'],
                                                                    request_data['role'], request_data['party_id'])
        return send_file(memory_file, attachment_filename=tar_file_name, as_attachment=True)


@manager.route('/component/output/data/table', methods=['post'])
@job_utils.job_server_routing()
def component_output_data_table():
    request_data = request.json
    data_views = query_data_view(**request_data)
    if data_views:
        return get_json_result(retcode=0, retmsg='success', data={'table_name': data_views[0].f_table_name,
                                                                  'table_namespace': data_views[0].f_table_namespace})
    else:
        return get_json_result(retcode=100, retmsg='No found table, please check if the parameters are correct')


# api using by task executor
@manager.route('/<job_id>/<component_name>/<task_id>/<role>/<party_id>/metric_data/save', methods=['POST'])
def save_metric_data(job_id, component_name, task_id, role, party_id):
    request_data = request.json
    tracker = Tracking(job_id=job_id, component_name=component_name, task_id=task_id, role=role, party_id=party_id)
    metrics = [deserialize_b64(metric) for metric in request_data['metrics']]
    tracker.save_metric_data(metric_namespace=request_data['metric_namespace'], metric_name=request_data['metric_name'],
                             metrics=metrics, job_level=request_data['job_level'])
    return get_json_result()


@manager.route('/<job_id>/<component_name>/<task_id>/<role>/<party_id>/metric_meta/save', methods=['POST'])
def save_metric_meta(job_id, component_name, task_id, role, party_id):
    request_data = request.json
    tracker = Tracking(job_id=job_id, component_name=component_name, task_id=task_id, role=role, party_id=party_id)
    metric_meta = deserialize_b64(request_data['metric_meta'])
    tracker.save_metric_meta(metric_namespace=request_data['metric_namespace'], metric_name=request_data['metric_name'],
                             metric_meta=metric_meta, job_level=request_data['job_level'])
    return get_json_result()


def get_component_output_data_table(task_data):
    check_request_parameters(task_data)
    tracker = Tracking(job_id=task_data['job_id'], component_name=task_data['component_name'],
                       role=task_data['role'], party_id=task_data['party_id'])
    job_dsl_parser = job_utils.get_job_dsl_parser_by_job_id(job_id=task_data['job_id'])
    if not job_dsl_parser:
        raise Exception('can not get dag parser, please check if the parameters are correct')
    component = job_dsl_parser.get_component_info(task_data['component_name'])
    if not component:
        raise Exception('can not found component, please check if the parameters are correct')
    output_dsl = component.get_output()
    output_data_dsl = output_dsl.get('data', [])
    # The current version will only have one data output.
    output_data_table = tracker.get_output_data_table(output_data_dsl[0] if output_data_dsl else 'component')
    return output_data_table


def get_component_output_data_line(src_key, src_value):
    have_data_label = False
    data_line = [src_key]
    if isinstance(src_value, Instance):
        if src_value.label is not None:
            data_line.append(src_value.label)
            have_data_label = True
        data_line.extend(data_utils.dataset_to_list(src_value.features))
    else:
        data_line.extend(data_utils.dataset_to_list(src_value))
    return data_line, have_data_label


def get_component_output_data_meta(output_data_table, have_data_label):
    # get meta
    output_data_meta = output_data_table.get_metas()
    schema = output_data_meta.get('schema', {})
    header = [schema.get('sid_name', 'sid')]
    if have_data_label:
        header.append(schema.get('label_name'))
    header.extend(schema.get('header', []))
    return header


def check_request_parameters(request_data):
    with DB.connection_context():
        if 'role' not in request_data and 'party_id' not in request_data:
            jobs = Job.select(Job.f_runtime_conf).where(Job.f_job_id == request_data.get('job_id', ''),
                                                        Job.f_is_initiator == 1)
            if jobs:
                job = jobs[0]
                job_runtime_conf = json_loads(job.f_runtime_conf)
                job_initiator = job_runtime_conf.get('initiator', {})
                role = job_initiator.get('role', '')
                party_id = job_initiator.get('party_id', 0)
                request_data['role'] = role
                request_data['party_id'] = party_id
