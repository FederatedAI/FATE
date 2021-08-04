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
from fate_flow.controller.engine_controller.engine import EngineABC
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.entity.run_status import TaskStatus
from fate_flow.entity.types import KillProcessRetCode
from fate_flow.operation.task_executor import TaskExecutor
from fate_flow.utils import job_utils
from fate_flow.db.db_models import Task
from fate_flow.entity.component_provider import ComponentProvider


class EggrollEngine(EngineABC):
    def run(self, task: Task, run_parameters, run_parameters_path, config_dir, log_dir, cwd_dir, **kwargs):
        process_cmd = [
            sys.executable,
            sys.modules[TaskExecutor.__module__].__file__,
            '-j', task.f_job_id,
            '-n', task.f_component_name,
            '-t', task.f_task_id,
            '-v', task.f_task_version,
            '-r', task.f_role,
            '-p', task.f_party_id,
            '-c', run_parameters_path,
            '--run_ip', RuntimeConfig.JOB_SERVER_HOST,
            '--job_server', '{}:{}'.format(RuntimeConfig.JOB_SERVER_HOST, RuntimeConfig.HTTP_PORT),
        ]

        provider = ComponentProvider(**task.f_provider_info)

        schedule_logger(task.f_job_id).info(f"task {task.f_task_id} {task.f_task_version} on {task.f_role} {task.f_party_id} executor subprocess is ready")
        p = job_utils.run_subprocess(job_id=task.f_job_id, config_dir=config_dir, process_cmd=process_cmd, extra_env=provider.env, log_dir=log_dir, cwd_dir=cwd_dir)
        return {"run_pid": p.pid}

    def kill(self, task):
        kill_status_code = job_utils.kill_task_executor_process(task)
        # session stop
        if kill_status_code == KillProcessRetCode.KILLED or task.f_status not in {TaskStatus.WAITING}:
            job_utils.start_session_stop(task)

    def is_alive(self, task):
        return job_utils.check_job_process(int(task.f_run_pid))