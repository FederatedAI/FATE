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

from fate_arch.common.base_utils import fate_uuid
from fate_arch import storage
from fate_flow.db.db_models import Job, DB
from fate_flow.manager.data_manager import delete_metric_data
from fate_flow.operation.job_tracker import Tracker
from fate_flow.settings import stat_logger, TEMP_DIRECTORY
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
    job_tracker = Tracker(job_id=request_data['job_id'], role=request_data['role'], party_id=request_data['party_id'])
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
    tracker = Tracker(job_id=request_data['job_id'], component_name=request_data['component_name'],
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
    tracker = Tracker(job_id=request_data['job_id'], component_name=request_data['component_name'],
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
    tracker = Tracker(job_id=request_data['job_id'], component_name=request_data['component_name'],
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
def component_output_model():
    request_data = request.json
    check_request_parameters(request_data)
    job_dsl, job_runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=request_data['job_id'],
                                                                                    role=request_data['role'],
                                                                                    party_id=request_data['party_id'])
    model_id = job_runtime_conf['job_parameters']['model_id']
    model_version = job_runtime_conf['job_parameters']['model_version']
    tracker = Tracker(job_id=request_data['job_id'], component_name=request_data['component_name'],
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
def component_output_data():
    request_data = request.json
    output_tables_meta = get_component_output_tables_meta(task_data=request_data)
    if not output_tables_meta:
        return get_json_result(retcode=0, retmsg='no data', data=[])
    output_data_list = []
    headers = []
    totals = []
    data_names = []
    for output_name, output_table_meta in output_tables_meta.items():
        output_data = []
        num = 100
        have_data_label = False
        is_str = False
        if output_table_meta:
            # part_of_data format: [(k, v)]
            for k, v in output_table_meta.get_part_of_data():
                if num == 0:
                    break
                data_line, have_data_label, is_str = get_component_output_data_line(src_key=k, src_value=v)
                output_data.append(data_line)
                num -= 1
            total = output_table_meta.get_count()
            output_data_list.append(output_data)
            data_names.append(output_name)
            totals.append(total)
        if output_data:
            header = get_component_output_data_schema(output_table_meta=output_table_meta, have_data_label=have_data_label, is_str=is_str)
            headers.append(header)
        else:
            headers.append(None)
    if len(output_data_list) == 1 and not output_data_list[0]:
        return get_json_result(retcode=0, retmsg='no data', data=[])
    return get_json_result(retcode=0, retmsg='success', data=output_data_list, meta={'header': headers, 'total': totals, 'names':data_names})


@manager.route('/component/output/data/download', methods=['get'])
def component_output_data_download():
    request_data = request.json
    output_tables_meta = get_component_output_tables_meta(task_data=request_data)
    limit = request_data.get('limit', -1)
    if not output_tables_meta:
        return error_response(response_code=500, retmsg='no data')
    if limit == 0:
        return error_response(response_code=500, retmsg='limit is 0')
    output_data_count = 0
    have_data_label = False

    output_data_file_list = []
    output_data_meta_file_list = []
    output_tmp_dir = os.path.join(os.getcwd(), 'tmp/{}'.format(fate_uuid()))
    for output_name, output_table_meta in output_tables_meta.items():
        is_str = False
        output_data_file_path = "{}/{}.csv".format(output_tmp_dir, output_name)
        os.makedirs(os.path.dirname(output_data_file_path), exist_ok=True)
        with open(output_data_file_path, 'w') as fw:
            with storage.Session.build(name=output_table_meta.get_name(), namespace=output_table_meta.get_namespace()) as storage_session:
                output_table = storage_session.get_table(name=output_table_meta.get_name(), namespace=output_table_meta.get_namespace())
                for k, v in output_table.collect():
                    data_line, have_data_label, is_str = get_component_output_data_line(src_key=k, src_value=v)
                    fw.write('{}\n'.format(','.join(map(lambda x: str(x), data_line))))
                    output_data_count += 1
                    if output_data_count == limit:
                        break

        if output_data_count:
            # get meta
            output_data_file_list.append(output_data_file_path)
            header = get_component_output_data_schema(output_table_meta=output_table_meta, have_data_label=have_data_label, is_str=is_str)
            output_data_meta_file_path = "{}/{}.meta".format(output_tmp_dir, output_name)
            output_data_meta_file_list.append(output_data_meta_file_path)
            with open(output_data_meta_file_path, 'w') as fw:
                json.dump({'header': header}, fw, indent=4)
            if request_data.get('head', True) and header:
                with open(output_data_file_path, 'r+') as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write('{}\n'.format(','.join(header)) + content)
    # tar
    memory_file = io.BytesIO()
    tar = tarfile.open(fileobj=memory_file, mode='w:gz')
    for index in range(0, len(output_data_file_list)):
        tar.add(output_data_file_list[index], os.path.relpath(output_data_file_list[index], output_tmp_dir))
        tar.add(output_data_meta_file_list[index], os.path.relpath(output_data_meta_file_list[index], output_tmp_dir))
    tar.close()
    memory_file.seek(0)
    output_data_file_list.extend(output_data_meta_file_list)
    for path in output_data_file_list:
        try:
            shutil.rmtree(os.path.dirname(path))
        except Exception as e:
            # warning
            stat_logger.warning(e)
        tar_file_name = 'job_{}_{}_{}_{}_output_data.tar.gz'.format(request_data['job_id'],
                                                                    request_data['component_name'],
                                                                    request_data['role'], request_data['party_id'])
        return send_file(memory_file, attachment_filename=tar_file_name, as_attachment=True)


@manager.route('/component/output/data/table', methods=['post'])
def component_output_data_table():
    output_data_infos = Tracker.query_output_data_infos(**request.json)
    if output_data_infos:
        return get_json_result(retcode=0, retmsg='success', data=[{'table_name': output_data_info.f_table_name,
                                                                  'table_namespace': output_data_info.f_table_namespace,
                                                                   "data_name": output_data_info.f_data_name
                                                                   } for output_data_info in output_data_infos])
    else:
        return get_json_result(retcode=100, retmsg='No found table, please check if the parameters are correct')


@manager.route('/component/summary/download', methods=['POST'])
def get_component_summary():
    request_data = request.json
    tracker = Tracker(job_id=request_data['job_id'], component_name=request_data['component_name'],
                      role=request_data['role'], party_id=request_data['party_id'])
    summary = tracker.get_component_summary()
    if summary:
        if request_data.get("filename"):
            temp_filepath = os.path.join(TEMP_DIRECTORY, request_data.get("filename"))
            with open(temp_filepath, "w") as fout:
                fout.write(json.dumps(summary, indent=4))
            return send_file(open(temp_filepath, "rb"), as_attachment=True,
                             attachment_filename=request_data.get("filename"))
        else:
            return get_json_result(data=summary)
    return error_response(500, "No component summary found, please check if arguments are specified correctly.")


@manager.route('/component/list', methods=['POST'])
def component_list():
    request_data = request.json
    parser = job_utils.get_job_dsl_parser_by_job_id(job_id=request_data.get('job_id'))
    if parser:
        return get_json_result(data={'components': list(parser.get_dsl().get('components').keys())})
    else:
        return get_json_result(retcode=100, retmsg='No job matched, please make sure the job id is valid.')


def get_component_output_tables_meta(task_data):
    check_request_parameters(task_data)
    tracker = Tracker(job_id=task_data['job_id'], component_name=task_data['component_name'],
                      role=task_data['role'], party_id=task_data['party_id'])
    job_dsl_parser = job_utils.get_job_dsl_parser_by_job_id(job_id=task_data['job_id'])
    if not job_dsl_parser:
        raise Exception('can not get dag parser, please check if the parameters are correct')
    component = job_dsl_parser.get_component_info(task_data['component_name'])
    if not component:
        raise Exception('can not found component, please check if the parameters are correct')
    output_data_table_infos = tracker.get_output_data_info()
    output_tables_meta = tracker.get_output_data_table(output_data_infos=output_data_table_infos)
    return output_tables_meta


def get_component_output_data_line(src_key, src_value):
    have_data_label = False
    data_line = [src_key]
    is_str = False
    if isinstance(src_value, Instance):
        if src_value.label is not None:
            data_line.append(src_value.label)
            have_data_label = True
        data_line.extend(data_utils.dataset_to_list(src_value.features))
    elif isinstance(src_value, str):
        data_line.extend([value for value in src_value.split(',')])
        is_str = True
    else:
        data_line.extend(data_utils.dataset_to_list(src_value))
    return data_line, have_data_label, is_str


def get_component_output_data_schema(output_table_meta, have_data_label, is_str=False):
    # get schema
    schema = output_table_meta.get_schema()
    if not schema:
         return None
    header = [schema.get('sid_name', 'sid')]
    if have_data_label:
        header.append(schema.get('label_name'))
    if is_str:
        header.extend([feature for feature in schema.get('header').split(',')])
    else:
        header.extend(schema.get('header', []))
    return header


def check_request_parameters(request_data):
    with DB.connection_context():
        if 'role' not in request_data and 'party_id' not in request_data:
            jobs = Job.select(Job.f_runtime_conf).where(Job.f_job_id == request_data.get('job_id', ''),
                                                        Job.f_is_initiator == True)
            if jobs:
                job = jobs[0]
                job_runtime_conf = job.f_runtime_conf
                job_initiator = job_runtime_conf.get('initiator', {})
                role = job_initiator.get('role', '')
                party_id = job_initiator.get('party_id', 0)
                request_data['role'] = role
                request_data['party_id'] = party_id
