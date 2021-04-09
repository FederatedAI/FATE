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
from fate_arch.common import file_utils, log, EngineType, profile
from fate_arch.common.base_utils import current_timestamp, timestamp_to_date
from fate_arch.common.log import schedule_logger, getLogger
from fate_arch import session
from fate_flow.entity.types import TaskStatus, ProcessRole, RunParameters
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.operation.job_tracker import Tracker
from fate_arch import storage
from fate_flow.utils import job_utils, schedule_utils
from fate_flow.scheduling_apps.client import ControllerClient
from fate_flow.scheduling_apps.client import TrackerClient
from fate_flow.db.db_models import TrackingOutputDataInfo, fill_db_model_object
from fate_arch.computing import ComputingEngine
from fate_flow.settings import WORK_MODE

LOGGER = getLogger()


class TaskExecutor(object):
    REPORT_TO_DRIVER_FIELDS = ["run_ip", "run_pid",
                               "party_status", "update_time", "end_time", "elapsed"]

    @classmethod
    def run_task(cls):
        task_info = {}
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                '-j', '--job_id', required=True, type=str, help="job id")
            parser.add_argument('-n', '--component_name', required=True, type=str,
                                help="component name")
            parser.add_argument('-t', '--task_id',
                                required=True, type=str, help="task id")
            parser.add_argument('-v', '--task_version',
                                required=True, type=int, help="task version")
            parser.add_argument(
                '-r', '--role', required=True, type=str, help="role")
            parser.add_argument('-p', '--party_id',
                                required=True, type=int, help="party id")
            parser.add_argument('-c', '--config', required=True,
                                type=str, help="task parameters")
            parser.add_argument('--run_ip', help="run ip", type=str)
            parser.add_argument('--job_server', help="job server", type=str)
            args = parser.parse_args()
            schedule_logger(args.job_id).info('enter task process')
            schedule_logger(args.job_id).info(args)
            # init function args
            if args.job_server:
                RuntimeConfig.init_config(JOB_SERVER_HOST=args.job_server.split(':')[0],
                                          HTTP_PORT=args.job_server.split(':')[1])
                RuntimeConfig.set_process_role(ProcessRole.EXECUTOR)
            job_id = args.job_id
            component_name = args.component_name
            task_id = args.task_id
            task_version = args.task_version
            role = args.role
            party_id = args.party_id
            executor_pid = os.getpid()
            task_info.update({
                "job_id": job_id,
                "component_name": component_name,
                "task_id": task_id,
                "task_version": task_version,
                "role": role,
                "party_id": party_id,
                "run_ip": args.run_ip,
                "run_pid": executor_pid
            })
            start_time = current_timestamp()
            job_conf = job_utils.get_job_conf(job_id, role)
            job_dsl = job_conf["job_dsl_path"]
            job_runtime_conf = job_conf["job_runtime_conf_path"]
            dsl_parser = schedule_utils.get_job_dsl_parser(dsl=job_dsl,
                                                           runtime_conf=job_runtime_conf,
                                                           train_runtime_conf=job_conf["train_runtime_conf_path"],
                                                           pipeline_dsl=job_conf["pipeline_dsl_path"]
                                                           )
            party_index = job_runtime_conf["role"][role].index(party_id)
            job_args_on_party = TaskExecutor.get_job_args_on_party(
                dsl_parser, job_runtime_conf, role, party_id)
            component = dsl_parser.get_component_info(
                component_name=component_name)
            component_parameters = component.get_role_parameters()
            component_parameters_on_party = component_parameters[role][
                party_index] if role in component_parameters else {}
            module_name = component.get_module()
            task_input_dsl = component.get_input()
            task_output_dsl = component.get_output()
            component_parameters_on_party['output_data_name'] = task_output_dsl.get(
                'data')
            task_parameters = RunParameters(
                **file_utils.load_json_conf(args.config))

            # use local work mode if it is different from the remote party
            task_parameters.work_mode = WORK_MODE

            job_parameters = task_parameters
            if job_parameters.assistant_role:
                TaskExecutor.monkey_patch()
            job_log_dir = os.path.join(job_utils.get_job_log_directory(
                job_id=job_id), role, str(party_id))
            task_log_dir = os.path.join(job_log_dir, component_name)
            log.LoggerFactory.set_directory(directory=task_log_dir, parent_log_dir=job_log_dir,
                                            append_to_parent_log=True, force=True)

            tracker = Tracker(job_id=job_id, role=role, party_id=party_id, component_name=component_name,
                              task_id=task_id,
                              task_version=task_version,
                              model_id=job_parameters.model_id,
                              model_version=job_parameters.model_version,
                              component_module_name=module_name,
                              job_parameters=job_parameters)
            tracker_client = TrackerClient(job_id=job_id, role=role, party_id=party_id,
                                           component_name=component_name,
                                           task_id=task_id,
                                           task_version=task_version,
                                           model_id=job_parameters.model_id,
                                           model_version=job_parameters.model_version,
                                           component_module_name=module_name,
                                           job_parameters=job_parameters)
            run_class_paths = component_parameters_on_party.get(
                'CodePath').split('/')
            run_class_package = '.'.join(
                run_class_paths[:-2]) + '.' + run_class_paths[-2].replace('.py', '')
            run_class_name = run_class_paths[-1]
            task_info["party_status"] = TaskStatus.RUNNING
            cls.report_task_update_to_driver(task_info=task_info)

            # init environment, process is shared globally
            RuntimeConfig.init_config(WORK_MODE=job_parameters.work_mode,
                                      COMPUTING_ENGINE=job_parameters.computing_engine,
                                      FEDERATION_ENGINE=job_parameters.federation_engine,
                                      FEDERATED_MODE=job_parameters.federated_mode)

            if RuntimeConfig.COMPUTING_ENGINE == ComputingEngine.EGGROLL:
                session_options = task_parameters.eggroll_run.copy()
            else:
                session_options = {}

            sess = session.Session(computing_type=job_parameters.computing_engine,
                                   federation_type=job_parameters.federation_engine)
            computing_session_id = job_utils.generate_session_id(
                task_id, task_version, role, party_id)
            sess.init_computing(
                computing_session_id=computing_session_id, options=session_options)
            federation_session_id = job_utils.generate_task_version_id(
                task_id, task_version)
            component_parameters_on_party["job_parameters"] = job_parameters.to_dict(
            )
            sess.init_federation(federation_session_id=federation_session_id,
                                 runtime_conf=component_parameters_on_party,
                                 service_conf=job_parameters.engines_address.get(EngineType.FEDERATION, {}))
            sess.as_default()

            schedule_logger().info('Run {} {} {} {} {} task'.format(
                job_id, component_name, task_id, role, party_id))
            schedule_logger().info("Component parameters on party {}".format(
                component_parameters_on_party))
            schedule_logger().info("Task input dsl {}".format(task_input_dsl))
            task_run_args = cls.get_task_run_args(job_id=job_id, role=role, party_id=party_id,
                                                  task_id=task_id,
                                                  task_version=task_version,
                                                  job_args=job_args_on_party,
                                                  job_parameters=job_parameters,
                                                  task_parameters=task_parameters,
                                                  input_dsl=task_input_dsl,
                                                  )
            if module_name in {"Upload", "Download", "Reader", "Writer"}:
                task_run_args["job_parameters"] = job_parameters
            run_object = getattr(importlib.import_module(
                run_class_package), run_class_name)()
            run_object.set_tracker(tracker=tracker_client)
            run_object.set_task_version_id(
                task_version_id=job_utils.generate_task_version_id(task_id, task_version))
            # add profile logs
            profile.profile_start()
            run_object.run(component_parameters_on_party, task_run_args)
            profile.profile_ends()
            output_data = run_object.save_data()
            if not isinstance(output_data, list):
                output_data = [output_data]
            for index in range(0, len(output_data)):
                data_name = task_output_dsl.get('data')[index] if task_output_dsl.get(
                    'data') else '{}'.format(index)
                persistent_table_namespace, persistent_table_name = tracker.save_output_data(
                    computing_table=output_data[index],
                    output_storage_engine=job_parameters.storage_engine,
                    output_storage_address=job_parameters.engines_address.get(EngineType.STORAGE, {}))
                if persistent_table_namespace and persistent_table_name:
                    tracker.log_output_data_info(data_name=data_name,
                                                 table_namespace=persistent_table_namespace,
                                                 table_name=persistent_table_name)
            output_model = run_object.export_model()
            # There is only one model output at the current dsl version.
            tracker.save_output_model(output_model,
                                      task_output_dsl['model'][0] if task_output_dsl.get('model') else 'default')
            task_info["party_status"] = TaskStatus.SUCCESS
        except Exception as e:
            traceback.print_exc()
            task_info["party_status"] = TaskStatus.FAILED
            schedule_logger().exception(e)
        finally:
            try:
                task_info["end_time"] = current_timestamp()
                task_info["elapsed"] = task_info["end_time"] - start_time
                cls.report_task_update_to_driver(task_info=task_info)
            except Exception as e:
                task_info["party_status"] = TaskStatus.FAILED
                traceback.print_exc()
                schedule_logger().exception(e)
        schedule_logger().info(
            'task {} {} {} start time: {}'.format(task_id, role, party_id, timestamp_to_date(start_time)))
        schedule_logger().info(
            'task {} {} {} end time: {}'.format(task_id, role, party_id, timestamp_to_date(task_info["end_time"])))
        schedule_logger().info(
            'task {} {} {} takes {}s'.format(task_id, role, party_id, int(task_info["elapsed"]) / 1000))
        schedule_logger().info(
            'Finish {} {} {} {} {} {} task {}'.format(job_id, component_name, task_id, task_version, role, party_id,
                                                      task_info["party_status"]))

        print('Finish {} {} {} {} {} {} task {}'.format(job_id, component_name, task_id, task_version, role, party_id,
                                                        task_info["party_status"]))
        return task_info

    @classmethod
    def get_job_args_on_party(cls, dsl_parser, job_runtime_conf, role, party_id):
        party_index = job_runtime_conf["role"][role].index(int(party_id))
        job_args = dsl_parser.get_args_input()
        job_args_on_party = job_args[role][party_index].get(
            'args') if role in job_args else {}
        return job_args_on_party

    @classmethod
    def get_task_run_args(cls, job_id, role, party_id, task_id, task_version, job_args, job_parameters: RunParameters, task_parameters: RunParameters,
                          input_dsl, filter_type=None, filter_attr=None, get_input_table=False):
        task_run_args = {}
        input_table = {}
        if 'idmapping' in role:
            return {}
        for input_type, input_detail in input_dsl.items():
            if filter_type and input_type not in filter_type:
                continue
            if input_type == 'data':
                this_type_args = task_run_args[input_type] = task_run_args.get(
                    input_type, {})
                for data_type, data_list in input_detail.items():
                    data_dict = {}
                    for data_key in data_list:
                        data_key_item = data_key.split('.')
                        data_dict[data_key_item[0]] = {data_type: []}
                    for data_key in data_list:
                        data_key_item = data_key.split('.')
                        search_component_name, search_data_name = data_key_item[0], data_key_item[1]
                        storage_table_meta = None
                        if search_component_name == 'args':
                            if job_args.get('data', {}).get(search_data_name).get('namespace', '') and job_args.get(
                                    'data', {}).get(search_data_name).get('name', ''):
                                storage_table_meta = storage.StorageTableMeta(
                                    name=job_args['data'][search_data_name]['name'], namespace=job_args['data'][search_data_name]['namespace'])
                        else:
                            tracker_client = TrackerClient(job_id=job_id, role=role, party_id=party_id,
                                                           component_name=search_component_name)
                            upstream_output_table_infos_json = tracker_client.get_output_data_info(
                                data_name=search_data_name)
                            if upstream_output_table_infos_json:
                                tracker = Tracker(job_id=job_id, role=role, party_id=party_id,
                                                  component_name=search_component_name)
                                upstream_output_table_infos = []
                                for _ in upstream_output_table_infos_json:
                                    upstream_output_table_infos.append(fill_db_model_object(
                                        Tracker.get_dynamic_db_model(TrackingOutputDataInfo, job_id)(), _))
                                output_tables_meta = tracker.get_output_data_table(
                                    output_data_infos=upstream_output_table_infos)
                                if output_tables_meta:
                                    storage_table_meta = output_tables_meta.get(
                                        search_data_name, None)
                        args_from_component = this_type_args[search_component_name] = this_type_args.get(
                            search_component_name, {})
                        if get_input_table and storage_table_meta:
                            input_table[data_key] = {'namespace': storage_table_meta.get_namespace(),
                                                     'name': storage_table_meta.get_name()}
                            computing_table = None
                        elif storage_table_meta:
                            LOGGER.info(
                                f"load computing table use {task_parameters.computing_partitions}")
                            computing_table = session.get_latest_opened().computing.load(
                                storage_table_meta.get_address(),
                                schema=storage_table_meta.get_schema(),
                                partitions=task_parameters.computing_partitions)
                        else:
                            computing_table = None

                        if not computing_table or not filter_attr or not filter_attr.get("data", None):
                            data_dict[search_component_name][data_type].append(
                                computing_table)
                            args_from_component[data_type] = data_dict[search_component_name][data_type]
                        else:
                            args_from_component[data_type] = dict(
                                [(a, getattr(computing_table, "get_{}".format(a))()) for a in filter_attr["data"]])
            elif input_type in ['model', 'isometric_model']:
                this_type_args = task_run_args[input_type] = task_run_args.get(
                    input_type, {})
                for dsl_model_key in input_detail:
                    dsl_model_key_items = dsl_model_key.split('.')
                    if len(dsl_model_key_items) == 2:
                        search_component_name, search_model_alias = dsl_model_key_items[
                            0], dsl_model_key_items[1]
                    elif len(dsl_model_key_items) == 3 and dsl_model_key_items[0] == 'pipeline':
                        search_component_name, search_model_alias = dsl_model_key_items[
                            1], dsl_model_key_items[2]
                    else:
                        raise Exception(
                            'get input {} failed'.format(input_type))
                    models = Tracker(job_id=job_id, role=role, party_id=party_id, component_name=search_component_name,
                                     model_id=job_parameters.model_id,
                                     model_version=job_parameters.model_version).get_output_model(
                        model_alias=search_model_alias)
                    this_type_args[search_component_name] = models
        if get_input_table:
            return input_table
        return task_run_args

    @classmethod
    def report_task_update_to_driver(cls, task_info):
        """
        Report task update to FATEFlow Server
        :param task_info:
        :return:
        """
        schedule_logger().info("report task {} {} {} {} to driver".format(
            task_info["task_id"],
            task_info["task_version"],
            task_info["role"],
            task_info["party_id"],
        ))
        ControllerClient.report_task(task_info=task_info)

    @classmethod
    def monkey_patch(cls):
        package_name = "monkey_patch"
        package_path = os.path.join(
            file_utils.get_python_base_directory(), "fate_flow", package_name)
        if not os.path.exists(package_path):
            return
        for f in os.listdir(package_path):
            f_path = os.path.join(
                file_utils.get_python_base_directory(), "fate_flow", package_name, f)
            if not os.path.isdir(f_path) or "__pycache__" in f_path:
                continue
            patch_module = importlib.import_module(
                "fate_flow." + package_name + '.' + f + '.monkey_patch')
            patch_module.patch_all()


if __name__ == '__main__':
    task_info = TaskExecutor.run_task()
    TaskExecutor.report_task_update_to_driver(task_info=task_info)
