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
import traceback

from arch.api import federation
from arch.api import session, Backend
from arch.api.utils import file_utils, log_utils
from arch.api.base.utils.store_type import StoreTypes
from arch.api.utils.core_utils import current_timestamp, get_lan_ip, timestamp_to_date
from arch.api.utils.log_utils import schedule_logger
from fate_flow.db.db_models import Task
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.manager.tracking_manager import Tracking
from fate_flow.settings import API_VERSION, SAVE_AS_TASK_INPUT_DATA_SWITCH, SAVE_AS_TASK_INPUT_DATA_IN_MEMORY
from fate_flow.utils import job_utils
from fate_flow.utils.api_utils import federated_api
from fate_flow.entity.constant_config import TaskStatus, ProcessRole


class TaskExecutor(object):
    @staticmethod
    def run_task():
        task = Task()
        task.f_create_time = current_timestamp()
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument('-j', '--job_id', required=True, type=str, help="job id")
            parser.add_argument('-n', '--component_name', required=True, type=str,
                                help="component name")
            parser.add_argument('-t', '--task_id', required=True, type=str, help="task id")
            parser.add_argument('-r', '--role', required=True, type=str, help="role")
            parser.add_argument('-p', '--party_id', required=True, type=str, help="party id")
            parser.add_argument('-c', '--config', required=True, type=str, help="task config")
            parser.add_argument('--processors_per_node', help="processors_per_node", type=int)
            parser.add_argument('--job_server', help="job server", type=str)
            args = parser.parse_args()
            schedule_logger(args.job_id).info('enter task process')
            schedule_logger(args.job_id).info(args)
            # init function args
            if args.job_server:
                RuntimeConfig.init_config(HTTP_PORT=args.job_server.split(':')[1])
                RuntimeConfig.set_process_role(ProcessRole.EXECUTOR)
            job_id = args.job_id
            component_name = args.component_name
            task_id = args.task_id
            role = args.role
            party_id = int(args.party_id)
            executor_pid = os.getpid()
            task_config = file_utils.load_json_conf(args.config)
            job_parameters = task_config['job_parameters']
            job_initiator = task_config['job_initiator']
            job_args = task_config['job_args']
            task_input_dsl = task_config['input']
            task_output_dsl = task_config['output']
            parameters = TaskExecutor.get_parameters(job_id, component_name, role, party_id)
            # parameters = task_config['parameters']
            module_name = task_config['module_name']
        except Exception as e:
            traceback.print_exc()
            schedule_logger().exception(e)
            task.f_status = TaskStatus.FAILED
            return
        try:
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
                               component_module_name=module_name)
            task.f_start_time = current_timestamp()
            task.f_run_ip = get_lan_ip()
            task.f_run_pid = executor_pid
            run_class_paths = parameters.get('CodePath').split('/')
            run_class_package = '.'.join(run_class_paths[:-2]) + '.' + run_class_paths[-2].replace('.py', '')
            run_class_name = run_class_paths[-1]
            task.f_status = TaskStatus.RUNNING
            TaskExecutor.sync_task_status(job_id=job_id, component_name=component_name, task_id=task_id, role=role,
                                          party_id=party_id, initiator_party_id=job_initiator.get('party_id', None),
                                          initiator_role=job_initiator.get('role', None),
                                          task_info=task.to_json())

            # init environment, process is shared globally
            RuntimeConfig.init_config(WORK_MODE=job_parameters['work_mode'],
                                      BACKEND=job_parameters.get('backend', 0))
            if args.processors_per_node and args.processors_per_node > 0 and RuntimeConfig.BACKEND == Backend.EGGROLL:
                session_options = {"eggroll.session.processors.per.node": args.processors_per_node}
            else:
                session_options = {}
            session.init(job_id=job_utils.generate_session_id(task_id, role, party_id),
                         mode=RuntimeConfig.WORK_MODE,
                         backend=RuntimeConfig.BACKEND,
                         options=session_options)
            federation.init(job_id=task_id, runtime_conf=parameters)

            schedule_logger().info('run {} {} {} {} {} task'.format(job_id, component_name, task_id, role, party_id))
            schedule_logger().info(parameters)
            schedule_logger().info(task_input_dsl)
            task_run_args = TaskExecutor.get_task_run_args(job_id=job_id, role=role, party_id=party_id,
                                                           task_id=task_id,
                                                           job_parameters=job_parameters, job_args=job_args,
                                                           input_dsl=task_input_dsl)
            run_object = getattr(importlib.import_module(run_class_package), run_class_name)()
            run_object.set_tracker(tracker=tracker)
            run_object.set_taskid(taskid=task_id)
            run_object.run(parameters, task_run_args)
            output_data = run_object.save_data()
            tracker.save_output_data_table(output_data, task_output_dsl.get('data')[0] if task_output_dsl.get('data') else 'component')
            output_model = run_object.export_model()
            # There is only one model output at the current dsl version.
            tracker.save_output_model(output_model, task_output_dsl['model'][0] if task_output_dsl.get('model') else 'default')
            task.f_status = TaskStatus.COMPLETE
        except Exception as e:
            task.f_status = TaskStatus.FAILED
            schedule_logger().exception(e)
        finally:
            sync_success = False
            try:
                task.f_end_time = current_timestamp()
                task.f_elapsed = task.f_end_time - task.f_start_time
                task.f_update_time = current_timestamp()
                TaskExecutor.sync_task_status(job_id=job_id, component_name=component_name, task_id=task_id, role=role,
                                              party_id=party_id,
                                              initiator_party_id=job_initiator.get('party_id', None),
                                              initiator_role=job_initiator.get('role', None),
                                              task_info=task.to_json())
                sync_success = True
            except Exception as e:
                traceback.print_exc()
                schedule_logger().exception(e)
        schedule_logger().info('task {} {} {} start time: {}'.format(task_id, role, party_id, timestamp_to_date(task.f_start_time)))
        schedule_logger().info('task {} {} {} end time: {}'.format(task_id, role, party_id, timestamp_to_date(task.f_end_time)))
        schedule_logger().info('task {} {} {} takes {}s'.format(task_id, role, party_id, int(task.f_elapsed)/1000))
        schedule_logger().info(
            'finish {} {} {} {} {} {} task'.format(job_id, component_name, task_id, role, party_id, task.f_status if sync_success else TaskStatus.FAILED))

        print('finish {} {} {} {} {} {} task'.format(job_id, component_name, task_id, role, party_id, task.f_status if sync_success else TaskStatus.FAILED))

    @staticmethod
    def get_task_run_args(job_id, role, party_id, task_id, job_parameters, job_args, input_dsl):
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

                                data_table = session.table(
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
                        # todo: If the same component has more than one identical input, save as is repeated
                        if SAVE_AS_TASK_INPUT_DATA_SWITCH:
                            if data_table:
                                schedule_logger().info("start save as task {} input data table {} {}".format(
                                    task_id,
                                    data_table.get_namespace(),
                                    data_table.get_name()))
                                origin_table_metas = data_table.get_metas()
                                origin_table_schema = data_table.schema
                                save_as_options = {"store_type": StoreTypes.ROLLPAIR_IN_MEMORY} if SAVE_AS_TASK_INPUT_DATA_IN_MEMORY else {}
                                data_table = data_table.save_as(
                                    namespace=job_utils.generate_session_id(task_id=task_id,
                                                                            role=role,
                                                                            party_id=party_id),
                                    name=data_table.get_name(), options=save_as_options)
                                data_table.save_metas(origin_table_metas)
                                data_table.schema = origin_table_schema
                                schedule_logger().info("save as task {} input data table to {} {} done".format(
                                    task_id,
                                    data_table.get_namespace(),
                                    data_table.get_name()))
                            else:
                                schedule_logger().info("pass save as task {} input data table, because the table is none".format(task_id))
                        else:
                            schedule_logger().info("pass save as task {} input data table, because the switch is off".format(task_id))
                        args_from_component[data_type] = data_table
            elif input_type in ['model', 'isometric_model']:
                this_type_args = task_run_args[input_type] = task_run_args.get(input_type, {})
                for dsl_model_key in input_detail:
                    dsl_model_key_items = dsl_model_key.split('.')
                    if len(dsl_model_key_items) == 2:
                        search_component_name, search_model_alias = dsl_model_key_items[0], dsl_model_key_items[1]
                    elif len(dsl_model_key_items) == 3 and dsl_model_key_items[0] == 'pipeline':
                        search_component_name, search_model_alias = dsl_model_key_items[1], dsl_model_key_items[2]
                    else:
                        raise Exception('get input {} failed'.format(input_type))
                    models = Tracking(job_id=job_id, role=role, party_id=party_id, component_name=search_component_name,
                                      model_id=job_parameters['model_id'],
                                      model_version=job_parameters['model_version']).get_output_model(
                        model_alias=search_model_alias)
                    this_type_args[search_component_name] = models
        return task_run_args

    @staticmethod
    def get_parameters(job_id, component_name, role, party_id):

        job_conf_dict = job_utils.get_job_conf(job_id)
        job_dsl_parser = job_utils.get_job_dsl_parser(dsl=job_conf_dict['job_dsl_path'],
                                                      runtime_conf=job_conf_dict['job_runtime_conf_path'],
                                                      train_runtime_conf=job_conf_dict['train_runtime_conf_path'])
        if job_dsl_parser:
            component = job_dsl_parser.get_component_info(component_name)
            parameters = component.get_role_parameters()
            role_index = parameters[role][0]['role'][role].index(party_id)
            return parameters[role][role_index]

    @staticmethod
    def sync_task_status(job_id, component_name, task_id, role, party_id, initiator_party_id, initiator_role, task_info, update=False):
        sync_success = True
        for dest_party_id in {party_id, initiator_party_id}:
            if party_id != initiator_party_id and dest_party_id == initiator_party_id:
                # do not pass the process id to the initiator
                task_info['f_run_ip'] = ''
            response = federated_api(job_id=job_id,
                                     method='POST',
                                     endpoint='/{}/schedule/{}/{}/{}/{}/{}/status'.format(
                                         API_VERSION,
                                         job_id,
                                         component_name,
                                         task_id,
                                         role,
                                         party_id),
                                     src_party_id=party_id,
                                     dest_party_id=dest_party_id,
                                     src_role=role,
                                     json_body=task_info,
                                     work_mode=RuntimeConfig.WORK_MODE)
            if response['retcode']:
                sync_success = False
                schedule_logger().exception('job {} role {} party {} synchronize task status failed'.format(job_id, role, party_id))
                break
        if not sync_success and not update:
            task_info['f_status'] = TaskStatus.FAILED
            TaskExecutor.sync_task_status(job_id, component_name, task_id, role, party_id, initiator_party_id,
                                          initiator_role, task_info, update=True)
        if update:
            raise Exception('job {} role {} party {} synchronize task status failed'.format(job_id, role, party_id))

if __name__ == '__main__':
    TaskExecutor.run_task()
