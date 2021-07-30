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
import os
import sys

from fate_arch.common.log import schedule_logger
from fate_flow.controller.engine_operation.base import BaseEngine
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.entity.types import KillProcessStatusCode, TaskStatus
from fate_flow.operation.task_executor import TaskExecutor
from fate_flow.utils import job_utils


class SparkEngine(BaseEngine):
    @staticmethod
    def run(job_id, component_name, task_id, task_version, role, party_id, task_parameters_path, run_parameters,
            task_info, **kwargs):
        if "SPARK_HOME" not in os.environ:
            raise EnvironmentError("SPARK_HOME not found")
        spark_home = os.environ["SPARK_HOME"]

        # additional configs
        spark_submit_config = run_parameters.spark_run

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
        task_log_dir = os.path.join(job_utils.get_job_log_directory(job_id=job_id), role, party_id, component_name)
        task_job_dir = os.path.join(job_utils.get_job_directory(job_id=job_id), role, party_id, component_name)
        schedule_logger(job_id).info(
            'job {} task {} {} on {} {} executor subprocess is ready'.format(job_id, task_id, task_version, role,
                                                                             party_id))
        task_dir = os.path.dirname(task_parameters_path)
        p = job_utils.run_subprocess(job_id=job_id, config_dir=task_dir, process_cmd=process_cmd, log_dir=task_log_dir,
                                     job_dir=task_job_dir)
        task_info["run_pid"] = p.pid
        return p

    @staticmethod
    def kill(task):
        kill_status_code = job_utils.kill_task_executor_process(task)
        # session stop
        if kill_status_code == KillProcessStatusCode.KILLED or task.f_status not in {TaskStatus.WAITING}:
            job_utils.start_session_stop(task)

    @staticmethod
    def is_alive(task):
        return job_utils.check_job_process(int(task.f_run_pid))