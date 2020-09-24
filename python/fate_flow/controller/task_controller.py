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

from fate_arch.common import FederatedCommunicationType
from fate_arch.common.log import schedule_logger
from fate_flow.db.db_models import Task
from fate_flow.operation.task_executor import TaskExecutor
from fate_flow.scheduler import FederatedScheduler
from fate_flow.entity.types import TaskStatus, EndStatus, KillProcessStatusCode
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.utils import job_utils
import os
from fate_flow.operation import JobSaver
from fate_arch.common.base_utils import json_dumps
from fate_arch.common import base_utils
from fate_flow.entity.types import RunParameters
from fate_flow.manager import ResourceManager
from fate_flow.operation import Tracker
from fate_arch.computing import ComputingEngine


class TaskController(object):
    INITIATOR_COLLECT_FIELDS = ["status", "party_status", "start_time", "update_time", "end_time", "elapsed"]

    @classmethod
    def create_task(cls, role, party_id, run_on, task_info):
        task_info["role"] = role
        task_info["party_id"] = party_id
        task_info["status"] = TaskStatus.WAITING
        task_info["party_status"] = TaskStatus.WAITING
        task_info["create_time"] = base_utils.current_timestamp()
        task_info["run_on"] = run_on
        if "task_id" not in task_info:
            task_info["task_id"] = job_utils.generate_task_id(job_id=task_info["job_id"], component_name=task_info["component_name"])
        if "task_version" not in task_info:
            task_info["task_version"] = 0
        JobSaver.create_task(task_info=task_info)

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
            'try to start job {} task {} {} on {} {} executor subprocess'.format(job_id, task_id, task_version, role, party_id))
        task_executor_process_start_status = False
        run_parameters = RunParameters(**task_parameters)
        task_info = {
            "job_id": job_id,
            "task_id": task_id,
            "task_version": task_version,
            "role": role,
            "party_id": party_id,
        }
        try:
            task_dir = os.path.join(job_utils.get_job_directory(job_id=job_id), role, party_id, component_name, task_id, task_version)
            os.makedirs(task_dir, exist_ok=True)
            task_parameters_path = os.path.join(task_dir, 'task_parameters.json')
            with open(task_parameters_path, 'w') as fw:
                fw.write(json_dumps(task_parameters))

            schedule_logger(job_id=job_id).info(f"use computing engine {run_parameters.computing_engine}")

            if run_parameters.computing_engine in {ComputingEngine.EGGROLL, ComputingEngine.STANDALONE}:
                process_cmd = [
                    sys.executable,  # the python executable path
                    sys.modules[TaskExecutor.__module__].__file__,
                    '-j', job_id,
                    '-n', component_name,
                    '-t', task_id,
                    '-v', task_version,
                    '-r', role,
                    '-p', party_id,
                    '-c', task_parameters_path,
                    '--processors_per_node', str(run_parameters.task_cores_per_node if run_parameters.task_cores_per_node else 0),
                    '--run_ip', RuntimeConfig.JOB_SERVER_HOST,
                    '--job_server', '{}:{}'.format(RuntimeConfig.JOB_SERVER_HOST, RuntimeConfig.HTTP_PORT),
                ]
            elif run_parameters.computing_engine == ComputingEngine.SPARK:
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
                    '--run_ip', RuntimeConfig.JOB_SERVER_HOST,
                    '--job_server', '{}:{}'.format(RuntimeConfig.JOB_SERVER_HOST, RuntimeConfig.HTTP_PORT),
                ])
                if run_parameters.task_nodes:
                    process_cmd.extend(["--num-executors", str(run_parameters.task_nodes)])
                if run_parameters.task_cores_per_node:
                    process_cmd.extend(["--executor-cores", str(run_parameters.task_cores_per_node)])
                if run_parameters.task_memory_per_node:
                    process_cmd.extend(["--executor-memory", f"{run_parameters.task_memory_per_node}m"])
            else:
                raise ValueError(f"${run_parameters.computing_engine} is not supported")

            task_log_dir = os.path.join(job_utils.get_job_log_directory(job_id=job_id), role, party_id, component_name)
            schedule_logger(job_id).info(
                'job {} task {} {} on {} {} executor subprocess is ready'.format(job_id, task_id, task_version, role, party_id))
            p = job_utils.run_subprocess(job_id=job_id, config_dir=task_dir, process_cmd=process_cmd, log_dir=task_log_dir)
            if p:
                task_info["party_status"] = TaskStatus.RUNNING
                task_info["run_pid"] = p.pid
                task_executor_process_start_status = True
            else:
                task_info["party_status"] = TaskStatus.FAILED
        except Exception as e:
            schedule_logger(job_id).exception(e)
            task_info["party_status"] = TaskStatus.FAILED
        finally:
            try:
                cls.update_task(task_info=task_info)
                cls.update_task_status(task_info=task_info)
            except Exception as e:
                schedule_logger(job_id).exception(e)
            schedule_logger(job_id).info(
                'job {} task {} {} on {} {} executor subprocess start {}'.format(job_id, task_id, task_version, role, party_id, "success" if task_executor_process_start_status else "failed"))

    @classmethod
    def update_task(cls, task_info):
        """
        Save to local database and then report to Initiator
        :param task_info:
        :return:
        """
        update_status = False
        try:
            update_status = JobSaver.update_task(task_info=task_info)
            cls.report_task_to_initiator(task_info=task_info)
        except Exception as e:
            schedule_logger(job_id=task_info["job_id"]).exception(e)
        finally:
            return update_status

    @classmethod
    def update_task_status(cls, task_info):
        update_status = JobSaver.update_task_status(task_info=task_info)
        if update_status and EndStatus.contains(task_info.get("status")):
            ResourceManager.return_task_resource(task_info=task_info)
        cls.report_task_to_initiator(task_info=task_info)
        return update_status

    @classmethod
    def report_task_to_initiator(cls, task_info):
        tasks = JobSaver.query_task(task_id=task_info["task_id"],
                                    task_version=task_info["task_version"],
                                    role=task_info["role"],
                                    party_id=task_info["party_id"])
        if tasks[0].f_federated_status_collect_type == FederatedCommunicationType.PUSH:
            FederatedScheduler.report_task_to_initiator(task=tasks[0])

    @classmethod
    def collect_task(cls, job_id, component_name, task_id, task_version, role, party_id):
        tasks = JobSaver.query_task(job_id=job_id, component_name=component_name, task_id=task_id, task_version=task_version, role=role, party_id=party_id)
        if tasks:
            return tasks[0].to_human_model_dict(only_primary_with=cls.INITIATOR_COLLECT_FIELDS)
        else:
            return None

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
        cls.update_task_status(task_info=task_info)
        cls.update_task(task_info=task_info)

    @classmethod
    def kill_task(cls, task: Task):
        kill_status = False
        try:
            # kill task executor
            kill_status_code = job_utils.kill_task_executor_process(task)
            # session stop
            if kill_status_code == KillProcessStatusCode.KILLED or task.f_status not in {TaskStatus.WAITING}:
                job_utils.start_session_stop(task)
        except Exception as e:
            schedule_logger(task.f_job_id).exception(e)
        finally:
            schedule_logger(task.f_job_id).info(
                'job {} task {} {} on {} {} process {} kill {}'.format(task.f_job_id, task.f_task_id,
                                                                       task.f_task_version,
                                                                       task.f_role,
                                                                       task.f_party_id,
                                                                       task.f_run_pid,
                                                                       'success' if kill_status else 'failed'))

    @classmethod
    def clean_task(cls, task, content_type):
        status = set()
        if content_type == "metrics":
            tracker = Tracker(job_id=task.f_job_id, role=task.f_role, party_id=task.f_party_id, task_id=task.f_task_id, task_version=task.f_task_version)
            status.add(tracker.clean_metrics())
        if len(status) == 1 and True in status:
            return True
        else:
            return False

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

