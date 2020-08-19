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

from fate_flow.db.db_models import Task
from fate_flow.entity.constant import TaskStatus, EndStatus, StatusSet, SchedulingStatusCode
from fate_flow.settings import API_VERSION, ALIGN_TASK_INPUT_DATA_PARTITION_SWITCH
from fate_flow.utils import job_utils
from fate_flow.utils.api_utils import federated_api
from fate_flow.scheduler.federated_scheduler import FederatedScheduler
from fate_flow.operation.job_saver import JobSaver
from fate_arch.common.log import schedule_logger
from fate_flow.manager.resource_manager import ResourceManager


class TaskScheduler(object):
    @classmethod
    def schedule(cls, job, dsl_parser):
        schedule_logger(job_id=job.f_job_id).info("Scheduling job {} tasks".format(job.f_job_id))
        tasks_group = JobSaver.get_tasks_asc(job_id=job.f_job_id, role=job.f_role, party_id=job.f_party_id)
        waiting_tasks = []
        for task_id, task in tasks_group.items():
            # update task status
            tasks = job_utils.query_task(task_id=task.f_task_id, task_version=task.f_task_version)
            tasks_party_status = [task.f_party_status for task in tasks]
            new_task_status = cls.calculate_multi_party_task_status(tasks_party_status=tasks_party_status)
            schedule_logger(job_id=task.f_job_id).info("Job {} task {} {} status is {}, calculate by task party status list: {}".format(task.f_job_id, task.f_task_id, task.f_task_version, new_task_status, tasks_party_status))
            task_status_have_update = False
            if new_task_status != task.f_status:
                task_status_have_update = True
                task.f_status = new_task_status
                FederatedScheduler.sync_task(job=job, task=task, update_fields=["status"])

            if task.f_status == TaskStatus.WAITING:
                waiting_tasks.append(task)
            elif task_status_have_update and EndStatus.contains(task.f_status):
                FederatedScheduler.stop_task(job=job, task=task, stop_status=task.f_status)

        scheduling_status_code = SchedulingStatusCode.NO_NEXT
        for waiting_task in waiting_tasks:
            # component = dsl_parser.get_component_info(component_name=waiting_task.f_component_name)
            # for component_name in component.get_upstream():
            for component in dsl_parser.get_upstream_dependent_components(component_name=waiting_task.f_component_name):
                dependent_task = tasks_group[job_utils.generate_task_id(job_id=job.f_job_id, component_name=component.get_name())]
                if dependent_task.f_status != TaskStatus.COMPLETE:
                    # can not start task
                    break
            else:
                # can start task
                scheduling_status_code = SchedulingStatusCode.HAVE_NEXT
                status_code = cls.start_task(job=job, task=waiting_task)
                tasks_group[waiting_task.f_task_id] = waiting_task
                if status_code == SchedulingStatusCode.NO_RESOURCE:
                    # Wait for the next round of scheduling
                    break
        schedule_logger(job_id=job.f_job_id).info("Finish scheduling job {} tasks".format(job.f_job_id))
        return scheduling_status_code, tasks_group.values()

    @classmethod
    def start_task(cls, job, task):
        schedule_logger(job_id=task.f_job_id).info("Try to start job {} task {} {} on {} {}".format(task.f_job_id, task.f_task_id, task.f_task_version, task.f_role, task.f_party_id))
        # TODO: apply for job resource
        apply_status = ResourceManager.apply_for_task_resource(task_info=task.to_human_model_dict(only_primary_with=["status"]))
        if not apply_status:
            return SchedulingStatusCode.NO_RESOURCE
        task.f_status = TaskStatus.RUNNING
        update_status = JobSaver.update_task(task_info=task.to_human_model_dict(only_primary_with=["status"]))
        if not update_status:
            # Another scheduler scheduling the task
            schedule_logger(job_id=task.f_job_id).info("Job {} task {} {} start on another scheduler".format(task.f_job_id, task.f_task_id, task.f_task_version))
            # Rollback
            task.f_status = TaskStatus.WAITING
            ResourceManager.return_task_resource(task_info=task.to_human_model_dict(only_primary_with=["status"]))
            return SchedulingStatusCode.PASS
        schedule_logger(job_id=task.f_job_id).info("Start job {} task {} {} on {} {}".format(task.f_job_id, task.f_task_id, task.f_task_version, task.f_role, task.f_party_id))
        task_parameters = {}
        #extra_task_parameters = TaskScheduler.align_task_parameters(job_id, job_parameters, job_initiator, job_args, component, task_id)
        #task_parameters.update(extra_task_parameters)
        task_parameters.update(job.f_runtime_conf["job_parameters"])
        FederatedScheduler.start_task(job=job, task=task, task_parameters=task_parameters)
        return SchedulingStatusCode.SUCCESS

    @staticmethod
    def align_task_parameters(job_id, job_parameters, job_initiator, job_args, component, task_id):
        parameters = component.get_role_parameters()
        component_name = component.get_name()
        extra_task_parameters = {'input_data_partition': 0}  # Large integers are not used
        for role, partys_parameters in parameters.items():
            for party_index in range(len(partys_parameters)):
                party_parameters = partys_parameters[party_index]
                if role in job_args:
                    party_job_args = job_args[role][party_index]['args']
                else:
                    party_job_args = {}
                dest_party_id = party_parameters.get('local', {}).get('party_id')
                if job_parameters.get('align_task_input_data_partition', ALIGN_TASK_INPUT_DATA_PARTITION_SWITCH):
                    response = federated_api(job_id=job_id,
                                             method='POST',
                                             endpoint='/{}/schedule/{}/{}/{}/{}/{}/input/args'.format(
                                                 API_VERSION,
                                                 job_id,
                                                 component_name,
                                                 task_id,
                                                 role,
                                                 dest_party_id),
                                             src_party_id=job_initiator['party_id'],
                                             dest_party_id=dest_party_id,
                                             src_role=job_initiator['role'],
                                             json_body={'job_parameters': job_parameters,
                                                        'job_args': party_job_args,
                                                        'input': component.get_input()},
                                             work_mode=job_parameters['work_mode'])
                    if response['retcode'] == 0:
                        for input_data in response.get('data', {}).get('data', {}).values():
                            for data_table_info in input_data.values():
                                if data_table_info and not isinstance(data_table_info, list):
                                    partitions = data_table_info['partitions']
                                    if extra_task_parameters['input_data_partition'] == 0 or partitions < extra_task_parameters['input_data_partition']:
                                        extra_task_parameters['input_data_partition'] = partitions
                    else:
                        raise Exception('job {} component {} align task parameters failed on {} {}'.format(job_id,
                                                                                                           component_name,
                                                                                                           role,
                                                                                                           dest_party_id))
        return extra_task_parameters

    @classmethod
    def calculate_multi_party_task_status(cls, tasks_party_status):
        # 1. all waiting
        # 2. have interrupt
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

    @classmethod
    def update_task_on_initiator(cls, initiator_task_template: Task, update_fields: list):
        tasks = JobSaver.query_task(task_id=initiator_task_template.f_task_id, task_version=initiator_task_template.f_task_version)
        if not tasks:
            raise Exception("Failed to update task status on initiator")
        task_info = initiator_task_template.to_human_model_dict(only_primary_with=update_fields)
        for field in update_fields:
            task_info[field] = getattr(initiator_task_template, "f_%s" % field)
        for task in tasks:
            task_info["role"] = task.f_role
            task_info["party_id"] = task.f_party_id
            JobSaver.update_task(task_info=task_info)
