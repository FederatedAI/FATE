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
from fate_arch import storage
from fate_flow.entity.types import RunParameters
from fate_flow.operation.job_saver import JobSaver
from fate_flow.operation.job_tracker import Tracker
from fate_flow.operation.task_executor import TaskExecutor
from fate_flow.utils.api_utils import get_json_result
from fate_flow.utils import detect_utils, job_utils, schedule_utils
from flask import request


@manager.route('/add', methods=['post'])
def table_add():
    request_data = request.json
    detect_utils.check_config(request_data, required_arguments=["engine", "address", "namespace", "name", ("head", (0, 1)), "id_delimiter"])
    address_dict = request_data.get('address')
    engine = request_data.get('engine')
    name = request_data.get('name')
    namespace = request_data.get('namespace')
    address = storage.StorageTableMeta.create_address(storage_engine=engine, address_dict=address_dict)
    in_serialized = request_data.get("in_serialized", 1 if engine in {storage.StorageEngine.STANDALONE, storage.StorageEngine.EGGROLL} else 0)
    destroy = (int(request_data.get("drop", 0)) == 1)
    data_table_meta = storage.StorageTableMeta(name=name, namespace=namespace)
    if data_table_meta:
        if destroy:
            data_table_meta.destroy_metas()
        else:
            return get_json_result(retcode=100,
                                   retmsg='The data table already exists.'
                                          'If you still want to continue uploading, please add the parameter -drop.'
                                          '1 means to add again after deleting the table')
    with storage.Session.build(storage_engine=engine, options=request_data.get("options")) as storage_session:
        storage_session.create_table(address=address, name=name, namespace=namespace, partitions=request_data.get('partitions', None),
                                     hava_head=request_data.get("head"), id_delimiter=request_data.get("id_delimiter"), in_serialized=in_serialized)
    return get_json_result(data={"table_name": name, "namespace": namespace})


@manager.route('/delete', methods=['post'])
def table_delete():
    request_data = request.json
    table_name = request_data.get('table_name')
    namespace = request_data.get('namespace')
    data = None
    with storage.Session.build(name=table_name, namespace=namespace) as storage_session:
        table = storage_session.get_table()
        if table:
            table.destroy()
            data = {'table_name': table_name, 'namespace': namespace}
    if data:
        return get_json_result(data=data)
    return get_json_result(retcode=101, retmsg='no find table')


@manager.route('/list', methods=['post'])
def get_job_table_list():
    detect_utils.check_config(config=request.json, required_arguments=['job_id', 'role', 'party_id'])
    jobs = JobSaver.query_job(**request.json)
    if jobs:
        job = jobs[0]
        tables = get_job_all_table(job)
        return get_json_result(data=tables)
    else:
        return get_json_result(retcode=101, retmsg='no find job')


@manager.route('/<table_func>', methods=['post'])
def table_api(table_func):
    config = request.json
    if table_func == 'table_info':
        table_key_count = 0
        table_partition = None
        table_schema = None
        table_name, namespace = config.get("name") or config.get("table_name"), config.get("namespace")
        table_meta = storage.StorageTableMeta(name=table_name, namespace=namespace)
        if table_meta:
            table_key_count = table_meta.get_count()
            table_partition = table_meta.get_partitions()
            table_schema = table_meta.get_schema()
            exist = 1
        else:
            exist = 0
        return get_json_result(data={"table_name": table_name,
                                     "namespace": namespace,
                                     "exist": exist,
                                     "count": table_key_count,
                                     "partition": table_partition,
                                     "schema": table_schema})
    else:
        return get_json_result()


def get_job_all_table(job):
    dsl_parser = schedule_utils.get_job_dsl_parser(dsl=job.f_dsl,
                                                   runtime_conf=job.f_runtime_conf,
                                                   train_runtime_conf=job.f_train_runtime_conf
                                                   )
    _, hierarchical_structure = dsl_parser.get_dsl_hierarchical_structure()
    component_table = {}
    try:
        component_output_tables = Tracker.query_output_data_infos(job_id=job.f_job_id, role=job.f_role,
                                                                  party_id=job.f_party_id)
    except:
        component_output_tables = []
    for component_name_list in hierarchical_structure:
        for component_name in component_name_list:
            component_table[component_name] = {}
            component_input_table = get_component_input_table(dsl_parser, job, component_name)
            component_table[component_name]['input'] = component_input_table
            component_table[component_name]['output'] = {}
            for output_table in component_output_tables:
                if output_table.f_component_name == component_name:
                    component_table[component_name]['output'][output_table.f_data_name] = \
                        {'name': output_table.f_table_name, 'namespace': output_table.f_table_namespace}
    return component_table


def get_component_input_table(dsl_parser, job, component_name):
    component = dsl_parser.get_component_info(component_name=component_name)
    module_name = get_component_module(component_name, job.f_dsl)
    if 'reader' in module_name.lower():
        component_parameters = component.get_role_parameters()
        return component_parameters[job.f_role][0]['ReaderParam']
    task_input_dsl = component.get_input()
    job_args_on_party = TaskExecutor.get_job_args_on_party(dsl_parser=dsl_parser,
                                                           job_runtime_conf=job.f_runtime_conf, role=job.f_role,
                                                           party_id=job.f_party_id)
    config = job_utils.get_job_parameters(job.f_job_id, job.f_role, job.f_party_id)
    task_parameters = RunParameters(**config)
    job_parameters = task_parameters
    component_input_table = TaskExecutor.get_task_run_args(job_id=job.f_job_id, role=job.f_role,
                                                           party_id=job.f_party_id,
                                                           task_id=None,
                                                           task_version=None,
                                                           job_args=job_args_on_party,
                                                           job_parameters=job_parameters,
                                                           task_parameters=task_parameters,
                                                           input_dsl=task_input_dsl,
                                                           get_input_table=True
                                                           )
    return component_input_table


def get_component_module(component_name, job_dsl):
    return job_dsl["components"][component_name]["module"].lower()
