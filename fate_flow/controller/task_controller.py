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
import sys

from arch.api.utils.log_utils import schedule_logger
from fate_flow.db.db_models import Task
from fate_flow.operation.task_executor import TaskExecutor
from fate_flow.scheduler.federated_scheduler import FederatedScheduler
from fate_flow.entity.constant import JobStatus, TaskSetStatus, TaskStatus
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.utils import job_utils, job_controller_utils
import os
from fate_flow.operation.job_saver import JobSaver
from arch.api.utils.core_utils import json_dumps
from fate_flow.entity.constant import Backend


class TaskController(object):
    @classmethod
    def start_task(cls, job_id, component_name, task_id, task_version, role, party_id, task_parameters):
        """
        Start task, update status and party status
        :param job_id:
        :param component_name:
        :param task_id:
        :param task_version:
        :param role:
        :param party_id:
        :param task_parameters:
        :return:
        """
        schedule_logger(job_id).info(
            'Try to start job {} task {} {} on {} {} executor subprocess'.format(job_id, task_id, task_version, role, party_id))
        task_executor_process_start_status = False
        try:
            task_info = {
                "job_id": job_id,
                "task_id": task_id,
                "task_version": task_version,
                "role": role,
                "party_id": party_id,
                "status": TaskStatus.RUNNING,
                "party_status": TaskStatus.RUNNING,
            }
            cls.update_task(task_info=task_info)
            task_dir = os.path.join(job_utils.get_job_directory(job_id=job_id), role, party_id, component_name, task_id, task_version)
            os.makedirs(task_dir, exist_ok=True)
            task_parameters_path = os.path.join(task_dir, 'task_parameters.json')
            with open(task_parameters_path, 'w') as fw:
                fw.write(json_dumps(task_parameters))

            backend = task_parameters.get("backend", Backend.EGGROLL)
            schedule_logger(job_id=job_id).info("use backend {}".format(Backend.EGGROLL))
            backend = Backend(backend)

            if backend.is_eggroll():
                process_cmd = [
                    'python3', sys.modules[TaskExecutor.__module__].__file__,
                    '-j', job_id,
                    '-n', component_name,
                    '-t', task_id,
                    '-v', task_version,
                    '-r', role,
                    '-p', party_id,
                    '-c', task_parameters_path,
                    '--processors_per_node', str(task_parameters.get("processors_per_node", 0)),
                    '--job_server', '{}:{}'.format(RuntimeConfig.JOB_SERVER_HOST, RuntimeConfig.HTTP_PORT),
                ]
            elif backend.is_spark():
                if "SPARK_HOME" not in os.environ:
                    raise EnvironmentError("SPARK_HOME not found")
                spark_home = os.environ["SPARK_HOME"]

                # additional configs
                spark_submit_config = task_parameters.get("spark_submit_config", dict())

                deploy_mode = spark_submit_config.get("deploy-mode", "client")
                if deploy_mode not in ["client"]:
                    raise ValueError(f"deploy mode {deploy_mode} not supported")

                spark_submit_cmd = os.path.join(spark_home, "bin/spark-submit")
                process_cmd = [spark_submit_cmd, f'--name={task_id}#{role}']
                for k, v in spark_submit_config.items():
                    if k != "conf":
                        process_cmd.append(f'--{k}={v}')
                if "conf" in spark_submit_config:
                    for ck, cv in spark_submit_config["conf"].items():
                        process_cmd.append(f'--conf')
                        process_cmd.append(f'{ck}={cv}')
                process_cmd.extend([
                    sys.modules[TaskExecutor.__module__].__file__,
                    '-j', job_id,
                    '-n', component_name,
                    '-t', task_id,
                    '-v', task_version,
                    '-r', role,
                    '-p', party_id,
                    '-c', task_parameters_path,
                    '--job_server', '{}:{}'.format(RuntimeConfig.JOB_SERVER_HOST, RuntimeConfig.HTTP_PORT),
                ])
            else:
                raise ValueError(f"${backend} supported")

            task_log_dir = os.path.join(job_utils.get_job_log_directory(job_id=job_id), role, party_id, component_name)
            schedule_logger(job_id).info(
                'Job {} task {} {} on {} {} executor subprocess is ready'.format(job_id, task_id, task_version, role, party_id))
            p = job_utils.run_subprocess(config_dir=task_dir, process_cmd=process_cmd, log_dir=task_log_dir)
            if p:
                task_executor_process_start_status = True
        except Exception as e:
            schedule_logger(job_id).exception(e)
        finally:
            schedule_logger(job_id).info(
                'Job {} task {} {} on {} {} executor subprocess start {}'.format(job_id, task_id, task_version, role, party_id, "success" if task_executor_process_start_status else "failed"))

    @classmethod
    def update_task(cls, task_info):
        """
        Save to local database and then report to Initiator
        :param task_info:
        :return:
        """
        JobSaver.update_task(task_info=task_info)
        tasks = job_utils.query_task(task_id=task_info["task_id"],
                                     task_version=task_info["task_version"],
                                     role=task_info["role"],
                                     party_id=task_info["party_id"])
        if len(tasks) == 1:
            FederatedScheduler.report_task_to_initiator(task=tasks[0])
        else:
            raise Exception("Found {} {} {} task on {} {}, error".format(len(tasks), task_info["task_id"], task_info["task_version"], task_info["role"], task_info["party_id"]))

    @classmethod
    def stop_task(cls, task, stop_status):
        """
        Try to stop the task, but the status depends on the final operation result
        :param task:
        :param stop_status:
        :return:
        """
        cls.kill_task(task=task)
        task_info = {
            "job_id": task.f_job_id,
            "task_id": task.f_task_id,
            "task_version": task.f_task_version,
            "role": task.f_role,
            "party_id": task.f_party_id,
            "party_status": stop_status
        }
        cls.update_task(task_info=task_info)

    @classmethod
    def kill_task(cls, task: Task):
        kill_status = False
        try:
            # kill task executor
            kill_status = job_utils.kill_task_executor_process(task)
            # session stop
            job_utils.start_session_stop(task)
        except Exception as e:
            schedule_logger(task.f_job_id).exception(e)
        finally:
            schedule_logger(task.f_job_id).info(
                'Job {} task {} {} on {} {} process {} kill {}'.format(task.f_job_id, task.f_task_id,
                                                                       task.f_task_version,
                                                                       task.f_role,
                                                                       task.f_party_id,
                                                                       task.f_run_pid,
                                                                       'success' if kill_status else 'failed'))

    @classmethod
    def query_task_input_args(cls, job_id, task_id, role, party_id, job_args, job_parameters, input_dsl, filter_type=None, filter_attr=None):
        task_run_args = TaskExecutor.get_task_run_args(job_id=job_id, role=role, party_id=party_id,
                                                       task_id=task_id,
                                                       job_args=job_args,
                                                       job_parameters=job_parameters,
                                                       task_parameters={},
                                                       input_dsl=input_dsl,
                                                       filter_type=filter_type,
                                                       filter_attr=filter_attr
                                                       )
        return task_run_args

