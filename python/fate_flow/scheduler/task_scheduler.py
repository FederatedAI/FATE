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
from fate_flow.entity.types import TaskStatus, EndStatus, StatusSet, SchedulingStatusCode, FederatedSchedulingStatusCode
from fate_flow.utils import job_utils
from fate_flow.scheduler import FederatedScheduler
from fate_flow.operation import JobSaver
from fate_arch.common.log import schedule_logger
from fate_flow.manager import ResourceManager


class TaskScheduler(object):
    @classmethod
    def schedule(cls, job, dsl_parser):
        schedule_logger(job_id=job.f_job_id).info("scheduling job {} tasks".format(job.f_job_id))
        initiator_tasks_group = JobSaver.get_tasks_asc(job_id=job.f_job_id, role=job.f_role, party_id=job.f_party_id)
        waiting_tasks = []
        for initiator_task in initiator_tasks_group.values():
            # collect all party task party status
            if job.f_runtime_conf["job_parameters"]["federated_status_collect_type"] == FederatedCommunicationType.PULL:
                tasks_on_all_party = JobSaver.query_task(task_id=initiator_task.f_task_id, task_version=initiator_task.f_task_version)
                tasks_status_on_all = set([task.f_status for task in tasks_on_all_party])
                if len(tasks_status_on_all) > 1 or TaskStatus.RUNNING in tasks_status_on_all:
                    cls.collect_task_of_all_party(job=job, task=initiator_task)
            new_task_status = cls.federated_task_status(job_id=initiator_task.f_job_id, task_id=initiator_task.f_task_id, task_version=initiator_task.f_task_version)
            task_status_have_update = False
            if new_task_status != initiator_task.f_status:
                task_status_have_update = True
                initiator_task.f_status = new_task_status
                FederatedScheduler.sync_task_status(job=job, task=initiator_task)

            if initiator_task.f_status == TaskStatus.WAITING:
                waiting_tasks.append(initiator_task)
            elif task_status_have_update and EndStatus.contains(initiator_task.f_status):
                FederatedScheduler.stop_task(job=job, task=initiator_task, stop_status=initiator_task.f_status)

        scheduling_status_code = SchedulingStatusCode.NO_NEXT
        for waiting_task in waiting_tasks:
            for component in dsl_parser.get_upstream_dependent_components(component_name=waiting_task.f_component_name):
                dependent_task = initiator_tasks_group[
                    JobSaver.task_key(task_id=job_utils.generate_task_id(job_id=job.f_job_id, component_name=component.get_name()),
                                      role=job.f_role,
                                      party_id=job.f_party_id
                                      )
                ]
                if dependent_task.f_status != TaskStatus.COMPLETE:
                    # can not start task
                    break
            else:
                # can start task
                scheduling_status_code = SchedulingStatusCode.HAVE_NEXT
                status_code = cls.start_task(job=job, task=waiting_task)
                if status_code == SchedulingStatusCode.NO_RESOURCE:
                    # Wait for the next round of scheduling
                    break
                elif status_code == SchedulingStatusCode.FAILED:
                    scheduling_status_code = SchedulingStatusCode.FAILED
                    break
        schedule_logger(job_id=job.f_job_id).info("finish scheduling job {} tasks".format(job.f_job_id))
        return scheduling_status_code, initiator_tasks_group.values()

    @classmethod
    def start_task(cls, job, task):
        schedule_logger(job_id=task.f_job_id).info("try to start job {} task {} {} on {} {}".format(task.f_job_id, task.f_task_id, task.f_task_version, task.f_role, task.f_party_id))
        apply_status = ResourceManager.apply_for_task_resource(task_info=task.to_human_model_dict(only_primary_with=["status"]))
        if not apply_status:
            return SchedulingStatusCode.NO_RESOURCE
        task.f_status = TaskStatus.RUNNING
        update_status = JobSaver.update_task_status(task_info=task.to_human_model_dict(only_primary_with=["status"]))
        if not update_status:
            # Another scheduler scheduling the task
            schedule_logger(job_id=task.f_job_id).info("job {} task {} {} start on another scheduler".format(task.f_job_id, task.f_task_id, task.f_task_version))
            # Rollback
            task.f_status = TaskStatus.WAITING
            ResourceManager.return_task_resource(task_info=task.to_human_model_dict(only_primary_with=["status"]))
            return SchedulingStatusCode.PASS
        schedule_logger(job_id=task.f_job_id).info("start job {} task {} {} on {} {}".format(task.f_job_id, task.f_task_id, task.f_task_version, task.f_role, task.f_party_id))
        task_parameters = {}
        task_parameters.update(job.f_runtime_conf["job_parameters"])
        status_code, response = FederatedScheduler.start_task(job=job, task=task, task_parameters=task_parameters)
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            return SchedulingStatusCode.SUCCESS
        else:
            return SchedulingStatusCode.FAILED

    @classmethod
    def collect_task_of_all_party(cls, job, task):
        status, federated_response = FederatedScheduler.collect_task(job=job, task=task)
        if status != FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).warning(f"collect task {task.f_task_id} {task.f_task_version} on {task.f_role} {task.f_party_id} failed")
            return
        for _role in federated_response.keys():
            for _party_id, party_response in federated_response[_role].items():
                JobSaver.update_task_status(task_info=party_response["data"])
                JobSaver.update_task(task_info=party_response["data"])

    @classmethod
    def federated_task_status(cls, job_id, task_id, task_version):
        tasks_on_all_party = JobSaver.query_task(task_id=task_id, task_version=task_version)
        tasks_party_status = [task.f_party_status for task in tasks_on_all_party]
        status = cls.calculate_multi_party_task_status(tasks_party_status)
        schedule_logger(job_id=job_id).info("job {} task {} {} status is {}, calculate by task party status list: {}".format(job_id, task_id, task_version, status, tasks_party_status))
        return status

    @classmethod
    def calculate_multi_party_task_status(cls, tasks_party_status):
        # 1. all waiting
        # 2. have end status, should be interrupted
        # 3. have running
        # 4. waiting + complete
        # 5. all the same end status
        tmp_status_set = set(tasks_party_status)
        if len(tmp_status_set) == 1:
            # 1 and 5
            return tmp_status_set.pop()
        else:
            # 2
            for status in sorted(EndStatus.status_list(), key=lambda s: StatusSet.get_level(status=s), reverse=True):
                if status == TaskStatus.COMPLETE:
                    continue
                if status in tmp_status_set:
                    return status
            # 3
            if TaskStatus.RUNNING in tmp_status_set or TaskStatus.COMPLETE in tmp_status_set:
                return StatusSet.RUNNING
            raise Exception("Calculate task status failed: {}".format(tasks_party_status))
