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

from fate_arch.common import base_utils
from fate_flow.utils.api_utils import federated_api
from fate_arch.common.log import schedule_logger
from fate_flow.entity.types import RetCode
from fate_flow.entity.run_status import FederatedSchedulingStatusCode
from fate_flow.db.db_models import Job, Task
from fate_flow.operation.job_saver import JobSaver
import threading
from fate_flow.utils import schedule_utils


class FederatedScheduler(object):
    """
    Send commands to party,
    Report info to initiator
    """

    # Task
    REPORT_TO_INITIATOR_FIELDS = ["party_status", "start_time", "update_time", "end_time", "elapsed"]

    # Job
    @classmethod
    def create_job(cls, job: Job):
        return cls.job_command(job=job, command="create", command_body=job.to_human_model_dict(), parallel=True)

    @classmethod
    def update_parameter(cls, job: Job, updated_parameters):
        return cls.job_command(job=job, command="parameter/update", command_body=updated_parameters, parallel=True)

    @classmethod
    def resource_for_job(cls, job, operation_type, specific_dest=None):
        schedule_logger(job_id=job.f_job_id).info(f"try to {operation_type} job {job.f_job_id} resource")
        status_code, response = cls.job_command(job=job, command=f"resource/{operation_type}", specific_dest=specific_dest)
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info(f"{operation_type} job {job.f_job_id} resource successfully")
        else:
            schedule_logger(job_id=job.f_job_id).info(f"{operation_type} job {job.f_job_id} resource failed")
        return status_code, response

    @classmethod
    def start_job(cls, job, command_body=None):
        return cls.job_command(job=job, command="start", command_body=command_body)

    @classmethod
    def align_args(cls, job, command_body):
        return cls.job_command(job=job, command="align", command_body=command_body)

    @classmethod
    def sync_job(cls, job, update_fields):
        sync_info = job.to_human_model_dict(only_primary_with=update_fields)
        schedule_logger(job_id=job.f_job_id).info("sync job {} info to all party".format(job.f_job_id))
        status_code, response = cls.job_command(job=job, command="update", command_body=sync_info)
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("sync job {} info to all party successfully".format(job.f_job_id))
        else:
            schedule_logger(job_id=job.f_job_id).info("sync job {} info to all party failed: \n{}".format(job.f_job_id, response))
        return status_code, response

    @classmethod
    def sync_job_status(cls, job):
        schedule_logger(job_id=job.f_job_id).info("job {} is {}, sync to all party".format(job.f_job_id, job.f_status))
        status_code, response = cls.job_command(job=job, command=f"status/{job.f_status}")
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("sync job {} status {} to all party success".format(job.f_job_id, job.f_status))
        else:
            schedule_logger(job_id=job.f_job_id).info("sync job {} status {} to all party failed: \n{}".format(job.f_job_id, job.f_status, response))
        return status_code, response

    @classmethod
    def save_pipelined_model(cls, job):
        schedule_logger(job_id=job.f_job_id).info("try to save job {} pipelined model".format(job.f_job_id))
        status_code, response = cls.job_command(job=job, command="model")
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("save job {} pipelined model success".format(job.f_job_id))
        else:
            schedule_logger(job_id=job.f_job_id).info("save job {} pipelined model failed:\n{}".format(job.f_job_id, response))
        return status_code, response

    @classmethod
    def stop_job(cls, job, stop_status):
        schedule_logger(job_id=job.f_job_id).info("try to stop job {}".format(job.f_job_id))
        job.f_status = stop_status
        status_code, response = cls.job_command(job=job, command="stop/{}".format(stop_status))
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("stop job {} success".format(job.f_job_id))
        else:
            schedule_logger(job_id=job.f_job_id).info("stop job {} failed:\n{}".format(job.f_job_id, response))
        return status_code, response

    @classmethod
    def request_stop_job(cls, job, stop_status, command_body=None):
        return cls.job_command(job=job, command="stop/{}".format(stop_status), dest_only_initiator=True, command_body=command_body)

    @classmethod
    def request_rerun_job(cls, job, command_body):
        return cls.job_command(job=job, command="rerun", command_body=command_body, dest_only_initiator=True)

    @classmethod
    def clean_job(cls, job):
        schedule_logger(job_id=job.f_job_id).info("try to clean job {}".format(job.f_job_id))
        status_code, response = cls.job_command(job=job, command="clean", command_body=job.f_runtime_conf_on_party["role"].copy())
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("clean job {} success".format(job.f_job_id))
        else:
            schedule_logger(job_id=job.f_job_id).info("clean job {} failed:\n{}".format(job.f_job_id, response))
        return status_code, response

    @classmethod
    def job_command(cls, job, command, command_body=None, dest_only_initiator=False, specific_dest=None, parallel=False):
        federated_response = {}
        job_parameters = job.f_runtime_conf_on_party["job_parameters"]
        if dest_only_initiator:
            dest_partis = [(job.f_initiator_role, [job.f_initiator_party_id])]
            api_type = "initiator"
        elif specific_dest:
            dest_partis = specific_dest.items()
            api_type = "party"
        else:
            dest_partis = job.f_roles.items()
            api_type = "party"
        threads = []
        for dest_role, dest_party_ids in dest_partis:
            federated_response[dest_role] = {}
            for dest_party_id in dest_party_ids:
                endpoint = f"/{api_type}/{job.f_job_id}/{dest_role}/{dest_party_id}/{command}"
                args = (job.f_job_id, job.f_role, job.f_party_id, dest_role, dest_party_id, endpoint, command_body, job_parameters["federated_mode"], federated_response)
                if parallel:
                    t = threading.Thread(target=cls.federated_command, args=args)
                    threads.append(t)
                    t.start()
                else:
                    cls.federated_command(*args)
        for thread in threads:
            thread.join()
        return cls.return_federated_response(federated_response=federated_response)

    @classmethod
    def create_task(cls, job, task):
        return cls.task_command(job=job, task=task, command="create", command_body=task.to_human_model_dict())

    @classmethod
    def start_task(cls, job, task):
        return cls.task_command(job=job, task=task, command="start", command_body={}, need_user=True)

    @classmethod
    def collect_task(cls, job, task):
        return cls.task_command(job=job, task=task, command="collect")

    @classmethod
    def sync_task(cls, job, task, update_fields):
        sync_info = task.to_human_model_dict(only_primary_with=update_fields)
        schedule_logger(job_id=task.f_job_id).info("sync job {} task {} {} info to all party".format(task.f_job_id, task.f_task_id, task.f_task_version))
        status_code, response = cls.task_command(job=job, task=task, command="update", command_body=sync_info)
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=task.f_job_id).info("sync job {} task {} {} info to all party successfully".format(task.f_job_id, task.f_task_id, task.f_task_version))
        else:
            schedule_logger(job_id=task.f_job_id).info("sync job {} task {} {} info to all party failed: \n{}".format(task.f_job_id, task.f_task_id, task.f_task_version, response))
        return status_code, response

    @classmethod
    def sync_task_status(cls, job, task):
        schedule_logger(job_id=task.f_job_id).info("job {} task {} {} is {}, sync to all party".format(task.f_job_id, task.f_task_id, task.f_task_version, task.f_status))
        status_code, response = cls.task_command(job=job, task=task, command=f"status/{task.f_status}")
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=task.f_job_id).info("sync job {} task {} {} status {} to all party success".format(task.f_job_id, task.f_task_id, task.f_task_version, task.f_status))
        else:
            schedule_logger(job_id=task.f_job_id).info("sync job {} task {} {} status {} to all party failed: \n{}".format(task.f_job_id, task.f_task_id, task.f_task_version, task.f_status, response))
        return status_code, response

    @classmethod
    def stop_task(cls, job, task, stop_status):
        schedule_logger(job_id=task.f_job_id).info("try to stop job {} task {} {}".format(task.f_job_id, task.f_task_id, task.f_task_version))
        task.f_status = stop_status
        status_code, response = cls.task_command(job=job, task=task, command="stop/{}".format(stop_status))
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("stop job {} task {} {} success".format(task.f_job_id, task.f_task_id, task.f_task_version))
        else:
            schedule_logger(job_id=job.f_job_id).info("stop job {} task {} {} failed:\n{}".format(task.f_job_id, task.f_task_id, task.f_task_version, response))
        return status_code, response

    @classmethod
    def clean_task(cls, job, task, content_type):
        schedule_logger(job_id=task.f_job_id).info("try to clean task {} {} {}".format(task.f_task_id, task.f_task_version, content_type))
        status_code, response = cls.task_command(job=job, task=task, command="clean/{}".format(content_type))
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("clean task {} {} {} successfully".format(task.f_task_id, task.f_task_version, content_type))
        else:
            schedule_logger(job_id=job.f_job_id).info("clean task {} {} {} failed:\n{}".format(task.f_task_id, task.f_task_version, content_type, response))
        return status_code, response

    @classmethod
    def task_command(cls, job, task, command, command_body=None, parallel=False, need_user=False):
        federated_response = {}
        job_parameters = job.f_runtime_conf_on_party["job_parameters"]
        tasks = JobSaver.query_task(task_id=task.f_task_id, task_version=task.f_task_version)
        threads = []
        for task in tasks:
            dest_role, dest_party_id = task.f_role, task.f_party_id
            federated_response[dest_role] = federated_response.get(dest_role, {})
            endpoint = f"/party/{task.f_job_id}/{task.f_component_name}/{task.f_task_id}/{task.f_task_version}/{dest_role}/{dest_party_id}/{command}"
            if need_user:
                command_body["user_id"] = job.f_user.get(dest_role, {}).get(str(dest_party_id), "")
                schedule_logger(job_id=job.f_job_id).info(f'user:{job.f_user}, dest_role:{dest_role}, dest_party_id:{dest_party_id}')
                schedule_logger(job_id=job.f_job_id).info(f'command_body: {command_body}')
            args = (job.f_job_id, job.f_role, job.f_party_id, dest_role, dest_party_id, endpoint, command_body, job_parameters["federated_mode"], federated_response)
            if parallel:
                t = threading.Thread(target=cls.federated_command, args=args)
                threads.append(t)
                t.start()
            else:
                cls.federated_command(*args)
        for thread in threads:
            thread.join()
        return cls.return_federated_response(federated_response=federated_response)

    @classmethod
    def federated_command(cls, job_id, src_role, src_party_id, dest_role, dest_party_id, endpoint, body, federated_mode, federated_response):
        st = base_utils.current_timestamp()
        schedule_logger(job_id).info(f"start sending {endpoint} federated command")
        try:
            response = federated_api(job_id=job_id,
                                     method='POST',
                                     endpoint=endpoint,
                                     src_role=src_role,
                                     src_party_id=src_party_id,
                                     dest_party_id=dest_party_id,
                                     json_body=body if body else {},
                                     federated_mode=federated_mode)
        except Exception as e:
            schedule_logger(job_id=job_id).exception(e)
            response = {
                "retcode": RetCode.FEDERATED_ERROR,
                "retmsg": "Federated schedule error, {}".format(e)
            }
        if response["retcode"] != RetCode.SUCCESS:
            schedule_logger(job_id=job_id).warning("an error occurred while {} the job to role {} party {}: \n{}".format(
                endpoint,
                dest_role,
                dest_party_id,
                response["retmsg"]
            ))
        federated_response[dest_role][dest_party_id] = response
        et = base_utils.current_timestamp()
        schedule_logger(job_id).info(f"send {endpoint} federated command use {et - st} ms")

    @classmethod
    def report_task_to_initiator(cls, task: Task):
        """
        :param task:
        :return:
        """
        if task.f_role != task.f_initiator_role and task.f_party_id != task.f_initiator_party_id:
            try:
                response = federated_api(job_id=task.f_job_id,
                                         method='POST',
                                         endpoint='/initiator/{}/{}/{}/{}/{}/{}/report'.format(
                                             task.f_job_id,
                                             task.f_component_name,
                                             task.f_task_id,
                                             task.f_task_version,
                                             task.f_role,
                                             task.f_party_id),
                                         src_party_id=task.f_party_id,
                                         dest_party_id=task.f_initiator_party_id,
                                         src_role=task.f_role,
                                         json_body=task.to_human_model_dict(only_primary_with=cls.REPORT_TO_INITIATOR_FIELDS),
                                         federated_mode=task.f_federated_mode)
            except Exception as e:
                schedule_logger(job_id=task.f_job_id).error(f"report task to initiator error: {e}")
                return False
            if response["retcode"] != RetCode.SUCCESS:
                retmsg = response["retmsg"]
                schedule_logger(job_id=task.f_job_id).error(f"report task to initiator error: {retmsg}")
                return False
            else:
                return True
        else:
            return False

    @classmethod
    def tracker_command(cls, job, request_data, command, json_body=None):
        job_parameters = job.f_runtime_conf_on_party["job_parameters"]
        response = federated_api(job_id=str(request_data['job_id']),
                                 method='POST',
                                 endpoint='/tracker/{}/{}/{}/{}/{}'.format(
                                     request_data['job_id'],
                                     request_data['component_name'],
                                     request_data['role'],
                                     request_data['party_id'],
                                     command),
                                 src_party_id=job.f_party_id,
                                 dest_party_id=request_data['party_id'],
                                 src_role=job.f_role,
                                 json_body=json_body if json_body else {},
                                 federated_mode=job_parameters["federated_mode"])
        return response

    # Utils
    @classmethod
    def return_federated_response(cls, federated_response):
        retcode_set = set()
        for dest_role in federated_response.keys():
            for party_id in federated_response[dest_role].keys():
                retcode_set.add(federated_response[dest_role][party_id]["retcode"])
        if len(retcode_set) == 1 and RetCode.SUCCESS in retcode_set:
            federated_scheduling_status_code = FederatedSchedulingStatusCode.SUCCESS
        elif RetCode.EXCEPTION_ERROR in retcode_set:
            federated_scheduling_status_code = FederatedSchedulingStatusCode.ERROR
        elif RetCode.SUCCESS in retcode_set:
            federated_scheduling_status_code = FederatedSchedulingStatusCode.PARTIAL
        else:
            federated_scheduling_status_code = FederatedSchedulingStatusCode.FAILED
        return federated_scheduling_status_code, federated_response
