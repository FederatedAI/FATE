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
from fate_arch.common import FederatedCommunicationType
from fate_arch.common.log import schedule_logger
from fate_flow.controller.engine_adapt import build_engine
from fate_flow.db.db_models import Task
from fate_flow.operation.task_executor import TaskExecutor
from fate_flow.scheduler.federated_scheduler import FederatedScheduler
from fate_flow.entity.types import TaskStatus, EndStatus
from fate_flow.utils import job_utils
import os
from fate_flow.operation.job_saver import JobSaver
from fate_arch.common.base_utils import json_dumps, current_timestamp
from fate_arch.common import base_utils
from fate_flow.entity.types import RunParameters
from fate_flow.manager.resource_manager import ResourceManager
from fate_flow.operation.job_tracker import Tracker
from fate_flow.utils.authentication_utils import PrivilegeAuth


class TaskController(object):
    INITIATOR_COLLECT_FIELDS = ["status", "party_status", "start_time", "update_time", "end_time", "elapsed"]

    @classmethod
    def create_task(cls, role, party_id, run_on_this_party, task_info):
        task_info["role"] = role
        task_info["party_id"] = party_id
        task_info["status"] = TaskStatus.WAITING
        task_info["party_status"] = TaskStatus.WAITING
        task_info["create_time"] = base_utils.current_timestamp()
        task_info["run_on_this_party"] = run_on_this_party
        if "task_id" not in task_info:
            task_info["task_id"] = job_utils.generate_task_id(job_id=task_info["job_id"], component_name=task_info["component_name"])
        if "task_version" not in task_info:
            task_info["task_version"] = 0
        JobSaver.create_task(task_info=task_info)

    @classmethod
    def start_task(cls, job_id, component_name, task_id, task_version, role, party_id, **kwargs):
        """
        Start task, update status and party status
        :param job_id:
        :param component_name:
        :param task_id:
        :param task_version:
        :param role:
        :param party_id:
        :return:
        """
        job_dsl = job_utils.get_job_dsl(job_id, role, party_id)
        PrivilegeAuth.authentication_component(job_dsl, src_party_id=kwargs.get('src_party_id'), src_role=kwargs.get('src_role'),
                                               party_id=party_id, component_name=component_name)

        schedule_logger(job_id).info(
            'try to start job {} task {} {} on {} {} executor subprocess'.format(job_id, task_id, task_version, role, party_id))
        task_executor_process_start_status = False
        task_info = {
            "job_id": job_id,
            "task_id": task_id,
            "task_version": task_version,
            "role": role,
            "party_id": party_id,
            "party_status": TaskStatus.RUNNING
        }
        is_failed = False
        try:
            task_dir = os.path.join(job_utils.get_job_directory(job_id=job_id), role, party_id, component_name, task_id, task_version)
            os.makedirs(task_dir, exist_ok=True)
            task_parameters_path = os.path.join(task_dir, 'task_parameters.json')
            run_parameters_dict = job_utils.get_job_parameters(job_id, role, party_id)
            run_parameters_dict["src_user"] = kwargs.get("src_user")
            with open(task_parameters_path, 'w') as fw:
                fw.write(json_dumps(run_parameters_dict))

            run_parameters = RunParameters(**run_parameters_dict)

            schedule_logger(job_id=job_id).info(f"use computing engine {run_parameters.computing_engine}")
            subprocess = True
            task_info["engine_conf"] = {"computing_engine": run_parameters.computing_engine}
            backend_engine = build_engine(run_parameters.computing_engine)
            status = backend_engine.run(job_id=job_id, component_name=component_name, task_id=task_id,
                                        task_version=task_version, role=role, party_id=party_id,
                                        task_parameters_path=task_parameters_path, run_parameters=run_parameters,
                                        task_info=task_info, user_name=kwargs.get("user_id"))
            if status:
                task_info["start_time"] = current_timestamp()
                task_executor_process_start_status = True
            else:
                is_failed = True
        except Exception as e:
            schedule_logger(job_id).exception(e)
            is_failed = True
        finally:
            try:
                cls.update_task(task_info=task_info)
                cls.update_task_status(task_info=task_info)
                if is_failed:
                    task_info["party_status"] = TaskStatus.FAILED
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
            cls.clean_task(job_id=task_info["job_id"],
                           task_id=task_info["task_id"],
                           task_version=task_info["task_version"],
                           role=task_info["role"],
                           party_id=task_info["party_id"],
                           content_type="table"
                           )
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
        kill_status = cls.kill_task(task=task)
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
        return kill_status

    @classmethod
    def kill_task(cls, task: Task):
        kill_status = False
        try:
            backend_engine = build_engine(task.f_engine_conf.get("computing_engine"))
            if backend_engine:
                backend_engine.kill(task)
        except Exception as e:
            schedule_logger(task.f_job_id).exception(e)
        else:
            kill_status = True
        finally:
            schedule_logger(task.f_job_id).info(
                'job {} task {} {} on {} {} process {} kill {}'.format(task.f_job_id, task.f_task_id,
                                                                       task.f_task_version,
                                                                       task.f_role,
                                                                       task.f_party_id,
                                                                       task.f_run_pid,
                                                                       'success' if kill_status else 'failed'))
            return kill_status

    @classmethod
    def clean_task(cls, job_id, task_id, task_version, role, party_id, content_type):
        status = set()
        if content_type == "metrics":
            tracker = Tracker(job_id=job_id, role=role, party_id=party_id, task_id=task_id, task_version=task_version)
            status.add(tracker.clean_metrics())
        elif content_type == "table":
            jobs = JobSaver.query_job(job_id=job_id, role=role, party_id=party_id)
            if jobs:
                job = jobs[0]
                job_parameters = RunParameters(**job.f_runtime_conf_on_party["job_parameters"])
                tracker = Tracker(job_id=job_id, role=role, party_id=party_id, task_id=task_id, task_version=task_version, job_parameters=job_parameters)
                status.add(tracker.clean_task(job.f_runtime_conf_on_party))
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

