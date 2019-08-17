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
import os

from arch.api import federation
from arch.api import storage
from arch.api.utils import file_utils, log_utils
from arch.api.utils.core import current_timestamp, get_lan_ip
from fate_flow.db.db_models import Task
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.manager.tracking import Tracking
from fate_flow.settings import API_VERSION, schedule_logger
from fate_flow.utils import job_utils
from fate_flow.utils.api_utils import federated_api
from fate_flow.entity.constant_config import TaskStatus


class TaskExecutor(object):
    @staticmethod
    def run_task():
        task = Task()
        task.f_create_time = current_timestamp()
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
            job_parameters = task_config.get('job_parameters', None)
            job_initiator = task_config.get('job_initiator', None)
            job_args = task_config.get('job_args', {})
            task_input_dsl = task_config.get('input', {})
            task_output_dsl = task_config.get('output', {})
            parameters = task_config.get('parameters', {})
            module_name = task_config.get('module_name', '')
        except Exception as e:
            schedule_logger.exception(e)
            task.f_status = TaskStatus.FAILED
            return
        try:
            # init environment
            RuntimeConfig.init_config(WORK_MODE=job_parameters['work_mode'])
            storage.init_storage(job_id=task_id, work_mode=RuntimeConfig.WORK_MODE)
            federation.init(job_id=task_id, runtime_conf=parameters)
            job_log_dir = os.path.join(job_utils.get_job_log_directory(job_id=job_id), role, str(party_id))
            task_log_dir = os.path.join(job_log_dir, component_name)
            log_utils.LoggerFactory.set_directory(directory=task_log_dir, parent_log_dir=job_log_dir,
                                                  append_to_parent_log=True, force=True)

            task.f_job_id = job_id
            task.f_component_name = component_name
            task.f_task_id = task_id
            task.f_role = role
            task.f_party_id = party_id
            task.f_operator = 'python_operator'
            tracker = Tracking(job_id=job_id, role=role, party_id=party_id, component_name=component_name,
                               task_id=task_id,
                               model_id=job_parameters['model_id'],
                               model_version=job_parameters['model_version'],
                               module_name=module_name)
            task.f_start_time = current_timestamp()
            task.f_run_ip = get_lan_ip()
            task.f_run_pid = os.getpid()
            run_class_paths = parameters.get('CodePath').split('/')
            run_class_package = '.'.join(run_class_paths[:-2]) + '.' + run_class_paths[-2].rstrip('.py')
            run_class_name = run_class_paths[-1]
            task_run_args = TaskExecutor.get_task_run_args(job_id=job_id, role=role, party_id=party_id,
                                                           job_parameters=job_parameters, job_args=job_args,
                                                           input_dsl=task_input_dsl)
            run_object = getattr(importlib.import_module(run_class_package), run_class_name)()
            run_object.set_tracker(tracker=tracker)
            run_object.set_taskid(taskid=task_id)
            task.f_status = TaskStatus.RUNNING
            TaskExecutor.sync_task_status(job_id=job_id, component_name=component_name, task_id=task_id, role=role,
                                          party_id=party_id, initiator_party_id=job_initiator.get('party_id', None),
                                          task_info=task.to_json())

            schedule_logger.info('run {} {} {} {} {} task'.format(job_id, component_name, task_id, role, party_id))
            schedule_logger.info(parameters)
            schedule_logger.info(task_input_dsl)
            run_object.run(parameters, task_run_args)
            if task_output_dsl:
                if task_output_dsl.get('data', []):
                    output_data = run_object.save_data()
                    tracker.save_output_data_table(output_data, task_output_dsl.get('data')[0])
                if task_output_dsl.get('model', []):
                    output_model = run_object.export_model()
                    # There is only one model output at the current dsl version.
                    tracker.save_output_model(output_model, task_output_dsl['model'][0])
            task.f_status = TaskStatus.SUCCESS
        except Exception as e:
            schedule_logger.exception(e)
            task.f_status = TaskStatus.FAILED
        finally:
            try:
                task.f_end_time = current_timestamp()
                task.f_elapsed = task.f_end_time - task.f_start_time
                task.f_update_time = current_timestamp()
                TaskExecutor.sync_task_status(job_id=job_id, component_name=component_name, task_id=task_id, role=role,
                                              party_id=party_id,
                                              initiator_party_id=job_initiator.get('party_id', None),
                                              task_info=task.to_json())
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

                                data_table = storage.table(
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
                for dsl_model_key in input_detail:
                    dsl_model_key_items = dsl_model_key.split('.')
                    if len(dsl_model_key_items) == 2:
                        search_component_name, search_model_name = dsl_model_key_items[0], dsl_model_key_items[1]
                    elif len(dsl_model_key_items) == 3 and dsl_model_key_items[0] == 'pipeline':
                        search_component_name, search_model_name = dsl_model_key_items[1], dsl_model_key_items[2]
                    else:
                        raise Exception('get input {} failed'.format(input_type))
                    models = Tracking(job_id=job_id, role=role, party_id=party_id, component_name=search_component_name,
                                      model_id=job_parameters['model_id'],
                                      model_version=job_parameters['model_version']).get_output_model(
                        model_name=search_model_name)
                    this_type_args[search_component_name] = models
        return task_run_args

    @staticmethod
    def sync_task_status(job_id, component_name, task_id, role, party_id, initiator_party_id, task_info):
        for dest_party_id in {party_id, initiator_party_id}:
            if party_id != initiator_party_id and dest_party_id == initiator_party_id:
                # do not pass the process id to the initiator
                task_info['f_run_ip'] = ''
            federated_api(job_id=job_id,
                          method='POST',
                          endpoint='/{}/job/{}/{}/{}/{}/{}/status'.format(
                              API_VERSION,
                              job_id,
                              component_name,
                              task_id,
                              role,
                              party_id),
                          src_party_id=party_id,
                          dest_party_id=dest_party_id,
                          json_body=task_info,
                          work_mode=RuntimeConfig.WORK_MODE)


if __name__ == '__main__':
    TaskExecutor.run_task()
