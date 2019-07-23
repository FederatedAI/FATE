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
import argparse
import importlib
import json
import os
import time

from arch.api import federation
from arch.api.proto import pipeline_pb2
from arch.api.utils import file_utils, log_utils, dtable_utils
from arch.api.utils.core import current_timestamp, json_dumps, base64_encode, json_loads, get_lan_ip
from fate_flow.db.db_models import Task, Job
from fate_flow.manager.queue_manager import JOB_QUEUE
from fate_flow.manager.tracking import Tracking
from fate_flow.settings import API_VERSION, schedule_logger
from fate_flow.storage.fate_storage import FateStorage
from fate_flow.utils import job_utils
from fate_flow.utils.api_utils import federated_api
from fate_flow.utils.job_utils import generate_job_id, save_job_conf, query_tasks, get_job_dsl_parser, run_subprocess


class JobController(object):
    task_executor_pool = None

    @staticmethod
    def init():
        pass

    @staticmethod
    def submit_job(job_data):
        job_id = generate_job_id()
        schedule_logger.info('submit job, job_id {}, body {}'.format(job_id, job_data))
        job_runtime_conf = job_data.get('job_runtime_conf', {})
        job_dsl = job_data.get('job_dsl', {})
        job_parameters = job_runtime_conf.get('job_parameters', {})
        if not job_parameters.get('model_key', None):
            model_key = '#'.join([dtable_utils.all_party_key(job_runtime_conf['role']), 'model'])
            job_parameters['model_key'] = model_key
        job_runtime_conf['job_parameters'] = job_parameters
        job_dsl_path, job_runtime_conf_path = save_job_conf(job_id=job_id,
                                                            job_dsl=job_dsl,
                                                            job_runtime_conf=job_runtime_conf)
        job_initiator = job_runtime_conf['initiator']
        initiator_role = job_initiator['role']
        initiator_party_id = job_initiator['party_id']
        job = Job()
        job.f_job_id = job_id
        job.f_roles = json_dumps(job_runtime_conf['role'])
        job.f_initiator_party_id = initiator_party_id
        job.f_dsl = json_dumps(job_dsl)
        job.f_runtime_conf = json_dumps(job_runtime_conf)
        job.f_run_ip = get_lan_ip()
        job.f_status = 'waiting'
        job.f_progress = 0
        job.f_create_time = current_timestamp()

        # save submit job info
        for role, partys in job_runtime_conf['role'].items():
            job.f_role = role
            for party_id in partys:
                job.f_party_id = party_id
                if role == initiator_role and party_id == initiator_party_id:
                    job.f_is_initiator = 1
                else:
                    job.f_is_initiator = 0
                federated_api(job_id=job_id,
                              method='POST',
                              url='/{}/job/{}/{}/{}/create'.format(
                                  API_VERSION,
                                  job_id,
                                  role,
                                  party_id),
                              src_party_id=initiator_party_id,
                              dest_party_id=party_id,
                              json_body=job.to_json())

        # generate model id
        model_version = job_id
        all_role_model_id = {}
        for _role, role_partys in job_runtime_conf['role'].items():
            all_role_model_id[_role] = []
            for _party_id in role_partys:
                all_role_model_id[_role].append(
                    Tracking.gen_party_model_id(job_parameters['model_key'], role=_role, party_id=_party_id))
        # push into queue
        JOB_QUEUE.put_event({
            'job_id': job_id,
            "job_dsl_path": job_dsl_path,
            "job_runtime_conf_path": job_runtime_conf_path
        }
        )
        schedule_logger.info(
            'submit job successfully, job id is {}, model key is {}'.format(job.f_job_id, job_parameters['model_key']))
        return job_id, job_dsl_path, job_runtime_conf_path, all_role_model_id, model_version

    @staticmethod
    def run_job(job_id, job_dsl_path, job_runtime_conf_path):
        dag = get_job_dsl_parser(job_dsl_path=job_dsl_path,
                                 job_runtime_conf_path=job_runtime_conf_path)
        job_runtime_conf = file_utils.load_json_conf(job_runtime_conf_path)
        job_parameters = job_runtime_conf.get('job_parameters', {})
        job_initiator = job_runtime_conf.get('initiator', {})
        job_args = dag.get_args_input()
        if not job_initiator:
            return False
        FateStorage.init_storage(job_id=job_id)
        job = Job()
        job.f_job_id = job_id
        job.f_start_time = current_timestamp()
        job.f_status = 'running'
        job.f_update_time = current_timestamp()
        JobController.update_job_status(job_id=job_id, roles=job_runtime_conf['role'],
                                        initiator_party_id=job_initiator['party_id'], job_info=job.to_json())

        top_level_task_status = set()
        components = dag.get_next_components(None)
        schedule_logger.info(
            '{} root components is {}'.format(job.f_job_id, [component.get_name() for component in components], None))
        for component in components:
            try:
                # run a component as task
                run_status = JobController.run_component(job_id, job_runtime_conf, job_parameters, job_initiator,
                                                         job_args, dag,
                                                         component)
            except Exception as e:
                schedule_logger.info(e)
                run_status = False
            top_level_task_status.add(run_status)
            if not run_status:
                break
        if len(top_level_task_status) == 2:
            job.f_status = 'partial'
        elif True in top_level_task_status:
            job.f_status = 'success'
        else:
            job.f_status = 'failed'
        job.f_end_time = current_timestamp()
        job.f_elapsed = job.f_end_time - job.f_start_time
        if job.f_status == 'success':
            job.f_progress = 100
        job.f_update_time = current_timestamp()
        JobController.update_job_status(job_id=job_id, roles=job_runtime_conf['role'],
                                        initiator_party_id=job_initiator['party_id'], job_info=job.to_json())
        JobController.finish_job(job_id=job_id, job_runtime_conf=job_runtime_conf)
        schedule_logger.info('job {} finished, status is {}'.format(job.f_job_id, job.f_status))

    @staticmethod
    def run_component(job_id, job_runtime_conf, job_parameters, job_initiator, job_args, dag, component):
        parameters = component.get_role_parameters()
        component_name = component.get_name()
        module_name = component.get_module()
        task_id = job_utils.generate_task_id(job_id=job_id, component_name=component_name)
        schedule_logger.info('run {} component {}'.format(job_id, component_name))
        for role, partys_parameters in parameters.items():
            for party_index in range(len(partys_parameters)):
                party_parameters = partys_parameters[party_index]
                if role in job_args:
                    party_job_args = job_args[role][party_index]['args']
                else:
                    party_job_args = {}
                dest_party_id = party_parameters.get('local', {}).get('party_id')
                federated_api(job_id=job_id,
                              method='POST',
                              url='/{}/job/{}/{}/{}/{}/{}/run'.format(
                                  API_VERSION,
                                  job_id,
                                  component_name,
                                  task_id,
                                  role,
                                  dest_party_id),
                              src_party_id=job_initiator['party_id'],
                              dest_party_id=dest_party_id,
                              json_body={'job_parameters': job_parameters,
                                         'job_initiator': job_initiator,
                                         'job_args': party_job_args,
                                         'parameters': party_parameters,
                                         'module_name': module_name,
                                         'input': component.get_input(),
                                         'output': component.get_output()})
        component_task_status = JobController.check_task_status(job_id=job_id, component=component)
        if component_task_status:
            task_success = True
        else:
            task_success = False
        schedule_logger.info(
            '{} component {} run {}'.format(job_id, component_name, 'success' if task_success else 'failed'))
        # update progress
        JobController.update_job_status(job_id=job_id, roles=job_runtime_conf['role'],
                                        initiator_party_id=job_initiator['party_id'],
                                        job_info=job_utils.update_job_progress(job_id=job_id, dag=dag,
                                                                               current_task_id=task_id).to_json())
        if task_success:
            next_components = dag.get_next_components(component_name)
            schedule_logger.info('{} component {} next components is {}'.format(job_id, component_name,
                                                                                [next_component.get_name() for
                                                                                 next_component in next_components]))
            for next_component in next_components:
                try:
                    schedule_logger.info(
                        '{} check component {} dependencies status'.format(job_id, next_component.get_name()))
                    dependencies_status = JobController.check_dependencies(job_id=job_id, dag=dag,
                                                                           component=next_component)
                    schedule_logger.info(
                        '{} component {} dependencies status is {}'.format(job_id, next_component.get_name(),
                                                                           dependencies_status))
                    if dependencies_status:
                        run_status = JobController.run_component(job_id, job_runtime_conf, job_parameters,
                                                                 job_initiator, job_args, dag,
                                                                 next_component)
                    else:
                        run_status = False
                except Exception as e:
                    schedule_logger.info(e)
                    run_status = False
                if not run_status:
                    return False
            return True
        else:
            return False

    @staticmethod
    def check_dependencies(job_id, dag, component):
        dependencies = dag.get_dependency().get('dependencies', {})
        if not dependencies:
            return False
        dependent_component_names = dependencies.get(component.get_name(), [])
        schedule_logger.info('{} component {} all dependent component: {}'.format(job_id, component.get_name(),
                                                                                  dependent_component_names))
        for dependent_component_name in dependent_component_names:
            dependent_component = dag.get_component_info(dependent_component_name)
            dependent_component_task_status = JobController.check_task_status(job_id, dependent_component)
            schedule_logger.info('{} component {} dependency {} status is {}'.format(job_id, component.get_name(),
                                                                                     dependent_component_name,
                                                                                     dependent_component_task_status))
            if not dependent_component_task_status:
                # dependency component run failed, break
                return False
        else:
            return True

    @staticmethod
    def check_task_status(job_id, component, interval=1):
        task_id = job_utils.generate_task_id(job_id=job_id, component_name=component.get_name())
        while True:
            try:
                status_collect = set()
                parameters = component.get_role_parameters()
                for _role, _partys_parameters in parameters.items():
                    for _party_parameters in _partys_parameters:
                        _party_id = _party_parameters.get('local', {}).get('party_id')
                        tasks = query_tasks(job_id=job_id, task_id=task_id, role=_role, party_id=_party_id)
                        if tasks:
                            task_status = tasks[0].f_status
                        else:
                            task_status = 'notRunning'
                        schedule_logger.info(
                            '{} component {} run on {} {} status is {}'.format(job_id, component.get_name(), _role,
                                                                               _party_id, task_status))
                        status_collect.add(task_status)
                if 'failed' in status_collect:
                    return False
                elif len(status_collect) == 1 and 'success' in status_collect:
                    return True
                else:
                    time.sleep(interval)
            except Exception as e:
                schedule_logger.exception(e)
                return False

    @staticmethod
    def start_task(job_id, component_name, task_id, role, party_id, task_config):
        task_dir = os.path.join(job_utils.get_job_directory(job_id=job_id), role, party_id, component_name)
        os.makedirs(task_dir, exist_ok=True)
        task_config_path = os.path.join(task_dir, 'task_config.json')
        with open(task_config_path, 'w') as fw:
            json.dump(task_config, fw)
        process_cmd = [
            'python3', __file__,
            '-j', job_id,
            '-n', component_name,
            '-t', task_id,
            '-r', role,
            '-p', party_id,
            '-c', task_config_path
        ]
        task_log_dir = os.path.join(job_utils.get_job_log_directory(job_id=job_id), role, party_id, component_name)
        p = run_subprocess(config_dir=task_dir, process_cmd=process_cmd, log_dir=task_log_dir)
        schedule_logger.info(
            'start {} {} {} {} task subprocess'.format(job_id, component_name, role, party_id, task_config))

    @staticmethod
    def run_task():
        task = Task()
        tracker = None
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument('-j', '--job_id', required=True, type=str, help="Specify a config json file path")
            parser.add_argument('-n', '--component_name', required=True, type=str,
                                help="Specify a config json file path")
            parser.add_argument('-t', '--task_id', required=True, type=str, help="Specify a config json file path")
            parser.add_argument('-r', '--role', required=True, type=str, help="Specify a config json file path")
            parser.add_argument('-p', '--party_id', required=True, type=str, help="Specify a config json file path")
            parser.add_argument('-c', '--config', required=True, type=str, help="Specify a config json file path")
            args = parser.parse_args()
            schedule_logger.info('enter task process')
            schedule_logger.info(args)
            # init function args
            job_id = args.job_id
            component_name = args.component_name
            task_id = args.task_id
            role = args.role
            party_id = int(args.party_id)
            task_config = file_utils.load_json_conf(args.config)
            request_url_without_host = task_config['request_url_without_host']
            job_parameters = task_config.get('job_parameters', None)
            job_initiator = task_config.get('job_initiator', None)
            job_args = task_config.get('job_args', {})
            task_input_dsl = task_config.get('input', {})
            task_output_dsl = task_config.get('output', {})
            parameters = task_config.get('parameters', {})
            module_name = task_config.get('module_name', '')
        except Exception as e:
            schedule_logger.exception(e)
            task.f_status = 'failed'
            return
        try:
            # init environment
            FateStorage.init_storage(job_id=job_id)
            federation.init(job_id=job_id, runtime_conf=parameters)
            job_log_dir = os.path.join(job_utils.get_job_log_directory(job_id=job_id), role, str(party_id))
            task_log_dir = os.path.join(job_log_dir, component_name)
            log_utils.LoggerFactory.set_directory(directory=task_log_dir, parent_log_dir=job_log_dir,
                                                  append_to_parent_log=True)

            task.f_job_id = job_id
            task.f_component_name = component_name
            task.f_task_id = task_id
            task.f_role = role
            task.f_party_id = party_id
            task.f_create_time = current_timestamp()
            tracker = Tracking(job_id=job_id, role=role, party_id=party_id, component_name=component_name,
                               task_id=task_id,
                               model_key=job_parameters['model_key'])
            task.f_start_time = current_timestamp()
            task.f_operator = 'python_operator'
            task.f_run_ip = get_lan_ip()
            task.f_run_pid = os.getpid()
            run_class_paths = parameters.get('CodePath').split('/')
            run_class_package = '.'.join(run_class_paths[:-2]) + '.' + run_class_paths[-2].rstrip('.py')
            run_class_name = run_class_paths[-1]
            task_run_args = JobController.get_task_run_args(job_id=job_id, role=role, party_id=party_id,
                                                            job_parameters=job_parameters, job_args=job_args,
                                                            input_dsl=task_input_dsl)
            run_object = getattr(importlib.import_module(run_class_package), run_class_name)()
            run_object.set_tracker(tracker=tracker)
            task.f_status = 'running'
            tracker.save_task(role=role, party_id=party_id, task_info=task.to_json(), create=True)

            schedule_logger.info('run {} {} {} {} {} task'.format(job_id, component_name, task_id, role, party_id))
            schedule_logger.info(parameters)
            schedule_logger.info(task_input_dsl)
            schedule_logger.info(task_run_args)
            run_object.run(parameters, task_run_args)
            if task_output_dsl:
                if task_output_dsl.get('data', {}):
                    output_data = run_object.save_data()
                    tracker.save_output_data_table(output_data, task_output_dsl.get('data')[0])
                if task_output_dsl.get('model', {}):
                    output_model = run_object.export_model()
                    tracker.save_output_model(output_model, module_name)
            task.f_status = 'success'
        except Exception as e:
            schedule_logger.exception(e)
            task.f_status = 'failed'
        finally:
            try:
                task.f_end_time = current_timestamp()
                task.f_elapsed = task.f_end_time - task.f_start_time
                task.f_update_time = current_timestamp()
                if tracker:
                    tracker.save_task(role=role, party_id=party_id, task_info=task.to_json(), create=True)
                # send task status to job initiator
                federated_api(job_id=job_id,
                              method='POST',
                              url=request_url_without_host.replace('run', 'status'),
                              src_party_id=task.f_party_id,
                              dest_party_id=job_initiator.get('party_id', None),
                              json_body=task.to_json())
            except Exception as e:
                schedule_logger.exception(e)
        schedule_logger.info(
            'finish {} {} {} {} {} {} task'.format(job_id, component_name, task_id, role, party_id, task.f_status))
        print('finish {} {} {} {} {} {} task'.format(job_id, component_name, task_id, role, party_id, task.f_status))

    @staticmethod
    def get_task_run_args(job_id, role, party_id, job_parameters, job_args, input_dsl):
        task_run_args = {}
        for input_type, input_detail in input_dsl.items():
            if input_type == 'data':
                this_type_args = task_run_args[input_type] = task_run_args.get(input_type, {})
                for data_type, data_list in input_detail.items():
                    for data_key in data_list:
                        data_key_item = data_key.split('.')
                        search_component_name, search_data_name = data_key_item[0], data_key_item[1]
                        if search_component_name == 'args':
                            if job_args.get('data', {}).get(search_data_name).get('namespace', '') and job_args.get(
                                    'data', {}).get(search_data_name).get('name', ''):

                                data_table = FateStorage.table(
                                    namespace=job_args['data'][search_data_name]['namespace'],
                                    name=job_args['data'][search_data_name]['name'])
                            else:
                                data_table = None
                        else:
                            data_table = Tracking(job_id=job_id, role=role, party_id=party_id,
                                                  component_name=search_component_name).get_output_data_table(
                                data_name=search_data_name)
                        args_from_component = this_type_args[search_component_name] = this_type_args.get(
                            search_component_name, {})
                        args_from_component[data_type] = data_table
            elif input_type in ['model', 'isometric_model']:
                this_type_args = task_run_args[input_type] = task_run_args.get(input_type, {})
                for model_key in input_detail:
                    model_key_items = model_key.split('.')
                    search_component_name, search_model_name = model_key_items[0], model_key_items[1]
                    models = Tracking(job_id=job_id, role=role, party_id=party_id, component_name=search_component_name,
                                      model_key=job_parameters['model_key']).get_output_model()
                    this_type_args[search_component_name] = models
        return task_run_args

    @staticmethod
    def kill_job(job_id):
        pass

    @staticmethod
    def update_job_status(job_id, roles, initiator_party_id, job_info):
        for role, partys in roles.items():
            job_info['f_role'] = role
            for party_id in partys:
                job_info['f_party_id'] = party_id
                federated_api(job_id=job_id,
                              method='POST',
                              url='/{}/job/{}/{}/{}/status'.format(
                                  API_VERSION,
                                  job_id,
                                  role,
                                  party_id),
                              src_party_id=initiator_party_id,
                              dest_party_id=party_id,
                              json_body=job_info)

    @staticmethod
    def job_status(job_id, role, party_id, job_info, create=False):
        job_tracker = Tracking(job_id=job_id, role=role, party_id=party_id)
        if create:
            save_job_conf(job_id=job_id,
                          job_dsl=json_loads(job_info['f_dsl']),
                          job_runtime_conf=json_loads(job_info['f_runtime_conf']))
            roles = json_loads(job_info['f_roles'])
            partner = {}
            show_role = {}
            is_initiator = job_info.get('f_is_initiator', 0)
            for _role, _role_party in roles.items():
                if is_initiator or _role == role:
                    show_role[_role] = show_role.get(_role, [])
                    for _party_id in _role_party:
                        if is_initiator or _party_id == party_id:
                            show_role[_role].append(_party_id)

                if _role != role:
                    partner[_role] = partner.get(_role, [])
                    partner[_role].extend(_role_party)
                else:
                    for _party_id in _role_party:
                        if _party_id != party_id:
                            partner[_role] = partner.get(_role, [])
                            partner[_role].append(_party_id)

            job_dsl_path, job_runtime_conf_path = job_utils.get_job_conf_path(job_id=job_id)
            dag = get_job_dsl_parser(job_dsl_path=job_dsl_path,
                                     job_runtime_conf_path=job_runtime_conf_path)
            job_args = dag.get_args_input()
            dataset = {}
            for _role, _role_party_args in job_args.items():
                if is_initiator or _role == role:
                    for _party_index in range(len(_role_party_args)):
                        _party_id = roles[_role][_party_index]
                        if is_initiator or _party_id == party_id:
                            dataset[_role] = dataset.get(_role, {})
                            dataset[_role][_party_id] = dataset[_role].get(_party_id, {})
                            for _data_type, _data_location in _role_party_args[_party_index]['args']['data'].items():
                                dataset[_role][_party_id][_data_type] = '{}.{}'.format(_data_location['namespace'],
                                                                                       _data_location['name'])
            job_tracker.log_job_view({'partner': partner, 'dataset': dataset, 'roles': show_role})
        job_tracker.save_job_info(role=role, party_id=party_id, job_info=job_info, create=create)

    @staticmethod
    def task_status(job_id, component_name, task_id, role, party_id, task_info):
        tracker = Tracking(job_id=job_id, role=role, party_id=party_id, component_name=component_name, task_id=task_id)
        tracker.save_task(role=role, party_id=party_id, task_info=task_info)
        schedule_logger.info(
            '{} component {} {} {} status {}'.format(job_id, component_name, role, party_id,
                                                     task_info.get('f_status', '')))

    @staticmethod
    def finish_job(job_id, job_runtime_conf):
        job_parameters = job_runtime_conf['job_parameters']
        job_initiator = job_runtime_conf['initiator']
        model_key_base64 = base64_encode(job_parameters['model_key'])
        for role, partys in job_runtime_conf['role'].items():
            for party_id in partys:
                # save pipeline
                federated_api(job_id=job_id,
                              method='POST',
                              url='/{}/job/{}/{}/{}/{}/save/pipeline'.format(
                                  API_VERSION,
                                  job_id,
                                  role,
                                  party_id,
                                  model_key_base64),
                              src_party_id=job_initiator['party_id'],
                              dest_party_id=party_id,
                              json_body={})
                # clean
                federated_api(job_id=job_id,
                              method='POST',
                              url='/{}/job/{}/{}/{}/clean'.format(
                                  API_VERSION,
                                  job_id,
                                  role,
                                  party_id),
                              src_party_id=job_initiator['party_id'],
                              dest_party_id=party_id,
                              json_body={})

    @staticmethod
    def save_pipeline(job_id, role, party_id, model_key):
        dsl_parser = job_utils.get_job_dsl_parser_by_job_id(job_id=job_id)
        predict_dsl = dsl_parser.get_predict_dsl(role=role)
        pipeline = pipeline_pb2.Pipeline()
        pipeline.inference_dsl = json_dumps(predict_dsl, byte=True)
        job_tracker = Tracking(job_id=job_id, role=role, party_id=party_id, model_key=model_key)
        job_tracker.save_output_model({'Pipeline': pipeline}, 'Pipeline')

    @staticmethod
    def clean_job(job_id, role, party_id):
        Tracking(job_id=job_id, role=role, party_id=party_id)


if __name__ == '__main__':
    JobController.run_task()
