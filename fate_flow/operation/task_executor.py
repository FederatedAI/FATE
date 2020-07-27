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
from arch.api.base.utils.store_type import StoreTypes
from arch.api.utils import file_utils, log_utils
from arch.api.utils.core_utils import current_timestamp, get_lan_ip, timestamp_to_date
from arch.api.utils.log_utils import schedule_logger
from fate_flow.entity.constant import TaskStatus, ProcessRole
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.operation.job_tracker import Tracker
from fate_flow.settings import API_VERSION, SAVE_AS_TASK_INPUT_DATA_SWITCH, SAVE_AS_TASK_INPUT_DATA_IN_MEMORY
from fate_flow.utils import job_utils
from fate_flow.utils import api_utils
from fate_flow.entity.constant import RetCode


class TaskExecutor(object):
    @classmethod
    def run_task(cls):
        task_info = {}
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument('-j', '--job_id', required=True, type=str, help="job id")
            parser.add_argument('-n', '--component_name', required=True, type=str,
                                help="component name")
            parser.add_argument('-t', '--task_id', required=True, type=str, help="task id")
            parser.add_argument('-v', '--task_version', required=True, type=str, help="task version")
            parser.add_argument('-r', '--role', required=True, type=str, help="role")
            parser.add_argument('-p', '--party_id', required=True, type=str, help="party id")
            parser.add_argument('-c', '--config', required=True, type=str, help="task parameters")
            parser.add_argument('--processors_per_node', help="processors_per_node", type=int)
            parser.add_argument('--job_server', help="job server", type=str)
            args = parser.parse_args()
            schedule_logger(args.job_id).info('enter task process')
            schedule_logger(args.job_id).info(args)
            # init function args
            if args.job_server:
                RuntimeConfig.init_config(JOB_SERVER_HOST=args.job_server.split(':')[0], HTTP_PORT=args.job_server.split(':')[1])
                RuntimeConfig.set_process_role(ProcessRole.EXECUTOR)
            job_id = args.job_id
            component_name = args.component_name
            task_id = args.task_id
            task_version = args.task_version
            role = args.role
            party_id = int(args.party_id)
            executor_pid = os.getpid()
            task_info.update({
                "job_id": job_id,
                "component_name": component_name,
                "task_id": task_id,
                "task_version": task_version,
                "role": role,
                "party_id": party_id,
                "run_ip": get_lan_ip(),
                "run_pid": executor_pid,
                "start_time": current_timestamp(),
            })
            job_conf = job_utils.get_job_conf(job_id)
            job_dsl = job_conf["job_dsl_path"]
            job_runtime_conf = job_conf["job_runtime_conf_path"]
            job_parameters = job_runtime_conf['job_parameters']
            job_initiator = job_runtime_conf['initiator']
            dsl_parser = job_utils.get_job_dsl_parser(dsl=job_dsl,
                                                      runtime_conf=job_runtime_conf,
                                                      train_runtime_conf=job_conf["train_runtime_conf_path"],
                                                      pipeline_dsl=job_conf["pipeline_dsl_path"]
                                                      )
            party_index = job_runtime_conf["role"][role].index(party_id)
            job_args = dsl_parser.get_args_input()
            job_args_on_party = job_args[role][party_index]["args"] if role in job_args else {}
            component = dsl_parser.get_component_info(component_name=component_name)
            component_parameters = component.get_role_parameters()
            component_parameters_on_party = component_parameters[role][party_index] if role in component_parameters else {}
            module_name = component.get_module()
            task_input_dsl = component.get_input()
            task_output_dsl = component.get_output()
            task_parameters = file_utils.load_json_conf(args.config)
            TaskExecutor.monkey_patch()
        except Exception as e:
            traceback.print_exc()
            schedule_logger().exception(e)
            task_info["party_status"] = TaskStatus.FAILED
            return
        try:
            job_log_dir = os.path.join(job_utils.get_job_log_directory(job_id=job_id), role, str(party_id))
            task_log_dir = os.path.join(job_log_dir, component_name)
            log_utils.LoggerFactory.set_directory(directory=task_log_dir, parent_log_dir=job_log_dir,
                                                  append_to_parent_log=True, force=True)

            tracker = Tracker(job_id=job_id, role=role, party_id=party_id, component_name=component_name,
                              task_id=task_id,
                              task_version=task_version,
                              model_id=job_parameters['model_id'],
                              model_version=job_parameters['model_version'],
                              component_module_name=module_name)
            run_class_paths = component_parameters_on_party.get('CodePath').split('/')
            run_class_package = '.'.join(run_class_paths[:-2]) + '.' + run_class_paths[-2].replace('.py', '')
            run_class_name = run_class_paths[-1]
            task_info["party_status"] = TaskStatus.RUNNING
            cls.report_task_to_server(task_info=task_info)

            # init environment, process is shared globally
            RuntimeConfig.init_config(WORK_MODE=job_parameters['work_mode'],
                                      BACKEND=job_parameters.get('backend', 0))
            if args.processors_per_node and args.processors_per_node > 0 and RuntimeConfig.BACKEND == Backend.EGGROLL:
                session_options = {"eggroll.session.processors.per.node": args.processors_per_node}
            else:
                session_options = {}
            session.init(job_id=job_utils.generate_session_id(task_id, task_version, role, party_id),
                         mode=RuntimeConfig.WORK_MODE,
                         backend=RuntimeConfig.BACKEND,
                         options=session_options)
            federation.init(job_id=job_utils.generate_federated_id(task_id, task_version), runtime_conf=component_parameters_on_party)

            schedule_logger().info('run {} {} {} {} {} task'.format(job_id, component_name, task_id, role, party_id))
            schedule_logger().info(component_parameters_on_party)
            schedule_logger().info(task_input_dsl)
            task_run_args = TaskExecutor.get_task_run_args(job_id=job_id, role=role, party_id=party_id,
                                                           task_id=task_id,
                                                           task_version=task_version,
                                                           job_args=job_args_on_party,
                                                           job_parameters=job_parameters,
                                                           task_parameters=task_parameters,
                                                           input_dsl=task_input_dsl,
                                                           if_save_as_task_input_data=job_parameters.get("save_as_task_input_data", SAVE_AS_TASK_INPUT_DATA_SWITCH)
                                                           )
            run_object = getattr(importlib.import_module(run_class_package), run_class_name)()
            run_object.set_tracker(tracker=tracker)
            run_object.set_taskid(taskid=job_utils.generate_federated_id(task_id, task_version))
            run_object.run(component_parameters_on_party, task_run_args)
            output_data = run_object.save_data()
            if not isinstance(output_data, list):
                output_data = [output_data]
            for index in range(0, len(output_data)):
                tracker.save_output_data_table(output_data[index], task_output_dsl.get('data')[index] if task_output_dsl.get('data') else 'component')
            output_model = run_object.export_model()
            # There is only one model output at the current dsl version.
            tracker.save_output_model(output_model, task_output_dsl['model'][0] if task_output_dsl.get('model') else 'default')
            task_info["party_status"] = TaskStatus.COMPLETE
        except Exception as e:
            task_info["party_status"] = TaskStatus.FAILED
            schedule_logger().exception(e)
        finally:
            try:
                task_info["end_time"] = current_timestamp()
                task_info["elapsed"] = task_info["end_time"] - task_info["start_time"]
                task_info["update_time"] = current_timestamp()
                cls.report_task_to_server(task_info=task_info)
            except Exception as e:
                task_info["party_status"] = TaskStatus.FAILED
                traceback.print_exc()
                schedule_logger().exception(e)
        schedule_logger().info('task {} {} {} start time: {}'.format(task_id, role, party_id, timestamp_to_date(task_info["start_time"])))
        schedule_logger().info('task {} {} {} end time: {}'.format(task_id, role, party_id, timestamp_to_date(task_info["end_time"])))
        schedule_logger().info('task {} {} {} takes {}s'.format(task_id, role, party_id, int(task_info["elapsed"])/1000))
        schedule_logger().info(
            'Finish {} {} {} {} {} {} task {}'.format(job_id, component_name, task_id, task_version, role, party_id, task_info["party_status"]))

        print('Finish {} {} {} {} {} {} task {}'.format(job_id, component_name, task_id, task_version, role, party_id, task_info["party_status"]))

    @staticmethod
    def get_task_run_args(job_id, role, party_id, task_id, task_version, job_args, job_parameters, task_parameters, input_dsl,
                          if_save_as_task_input_data, filter_type=None, filter_attr=None):
        task_run_args = {}
        for input_type, input_detail in input_dsl.items():
            if filter_type and input_type not in filter_type:
                continue
            if input_type == 'data':
                this_type_args = task_run_args[input_type] = task_run_args.get(input_type, {})
                for data_type, data_list in input_detail.items():
                    data_dict = {}
                    for data_key in data_list:
                        data_key_item = data_key.split('.')
                        data_dict[data_key_item[0]] = {data_type: []}
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
                            data_table = Tracker(job_id=job_id, role=role, party_id=party_id,
                                                 component_name=search_component_name).get_output_data_table(
                                data_name=search_data_name)
                        args_from_component = this_type_args[search_component_name] = this_type_args.get(
                            search_component_name, {})
                        # todo: If the same component has more than one identical input, save as is repeated
                        if if_save_as_task_input_data:
                            if data_table:
                                schedule_logger().info("start save as task {} input data table {} {}".format(
                                    task_id,
                                    data_table.get_namespace(),
                                    data_table.get_name()))
                                origin_table_metas = data_table.get_metas()
                                origin_table_schema = data_table.schema
                                save_as_options = {"store_type": StoreTypes.ROLLPAIR_IN_MEMORY} if SAVE_AS_TASK_INPUT_DATA_IN_MEMORY else {}
                                # TODO: data table name concat random string
                                data_table = data_table.save_as(
                                    namespace=job_utils.generate_session_id(task_id=task_id,
                                                                            task_version=task_version,
                                                                            role=role,
                                                                            party_id=party_id),
                                    name=data_table.get_name(),
                                    partition=task_parameters['input_data_partition'] if task_parameters.get('input_data_partition', 0) > 0 else data_table.get_partitions(),
                                    options=save_as_options)
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
                        if not data_table or not filter_attr or not filter_attr.get("data", None):
                            data_dict[search_component_name][data_type].append(data_table)
                            args_from_component[data_type] = data_dict[search_component_name][data_type]
                        else:
                            args_from_component[data_type] = dict([(a, getattr(data_table, "get_{}".format(a))()) for a in filter_attr["data"]])
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
                    models = Tracker(job_id=job_id, role=role, party_id=party_id, component_name=search_component_name,
                                     model_id=job_parameters['model_id'],
                                     model_version=job_parameters['model_version']).get_output_model(
                        model_alias=search_model_alias)
                    this_type_args[search_component_name] = models
        return task_run_args

    @classmethod
    def report_task_to_server(cls, task_info):
        """
        Report task info to FATEFlow Server
        :param task_info:
        :return:
        """
        try_times = 3
        for t in range(try_times):
            try:
                response = api_utils.local_api(
                                         job_id=task_info["job_id"],
                                         method='POST',
                                         endpoint='/{}/control/{}/{}/{}/{}/{}/{}/update'.format(
                                             API_VERSION,
                                             task_info["job_id"],
                                             task_info["component_name"],
                                             task_info["task_id"],
                                             task_info["task_version"],
                                             task_info["role"],
                                             task_info["party_id"]
                                         ),
                                         json_body=task_info)
                if response["retcode"] == 0:
                    break
            except Exception as e:
                response = {
                    "retcode": RetCode.CONNECTION_ERROR,
                    "retmsg": "Report error, {}".format(e)
                }
            if response["retcode"] != 0:
                schedule_logger().warning("report task {} {} {} {} to server failed, try again: \n{}".format(
                    task_info["task_id"],
                    task_info["task_version"],
                    task_info["role"],
                    task_info["party_id"],
                    response["retmsg"]
                ))
        else:
            raise Exception("report task {} {} {} {} to server failed".format(
                task_info["task_id"],
                task_info["task_version"],
                task_info["role"],
                task_info["party_id"],
            ))

    @staticmethod
    def monkey_patch():
        package_name = "monkey_patch"
        package_path = os.path.join(file_utils.get_project_base_directory(), package_name)
        for f in os.listdir(package_path):
            if not os.path.isdir(f) or f == "__pycache__":
                continue
            patch_module = importlib.import_module(package_name + '.' + f + '.monkey_patch')
            patch_module.patch_all()


if __name__ == '__main__':
    TaskExecutor.run_task()
