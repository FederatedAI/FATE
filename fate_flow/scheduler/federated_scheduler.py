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

from fate_flow.settings import API_VERSION
from fate_flow.utils.api_utils import federated_api
from arch.api.utils.core_utils import json_dumps
from arch.api.utils.log_utils import schedule_logger
from fate_flow.entity.constant import RetCode, FederatedSchedulingStatusCode
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.db.db_models import Job, TaskSet, Task


class FederatedScheduler(object):
    """
    The Scheduler sends commands to all,
    All report status to Scheduler
    """
    @classmethod
    def create_job(cls, job: Job):
        job_info = {}
        for k, v in job.to_json().items():
            job_info[k.lstrip("f_")] = v
        retcode, response = cls.job_command(job=job, command="create", command_body=job_info)
        if retcode != 0:
            raise Exception("Create job failed: {}".format(json_dumps(response, indent=4)))

    @classmethod
    def initialize_job(cls, job):
        return cls.job_command(job=job, command="initialize")

    @classmethod
    def check_job(cls, job):
        return cls.job_command(job=job, command="check")

    @classmethod
    def cancel_ready(cls, job):
        # TODO: sync job tag cancel_ready
        return True

    @classmethod
    def start_job(cls, job):
        return cls.job_command(job=job, command="start")

    @classmethod
    def save_pipelined_model(cls, job):
        return cls.job_command(job=job, command="model")

    @classmethod
    def stop_job(cls, job, stop_status):
        return cls.job_command(job=job, command="stop/{}".format(stop_status))

    @classmethod
    def cancel_job(cls, job):
        return cls.job_command(job=job, command="cancel")

    @classmethod
    def clean_job(cls, job):
        return cls.job_command(job=job, command="clean")

    @classmethod
    def job_command(cls, job, command, command_body=None):
        federated_response = {}
        roles, job_initiator = job.f_runtime_conf["role"], job.f_runtime_conf['initiator']
        for dest_role, dest_party_ids in roles.items():
            federated_response[dest_role] = {}
            for dest_party_id in dest_party_ids:
                try:
                    response = federated_api(job_id=job.f_job_id,
                                                  method='POST',
                                                  endpoint='/{}/schedule/{}/{}/{}/{}'.format(
                                                      API_VERSION,
                                                      job.f_job_id,
                                                      dest_role,
                                                      dest_party_id,
                                                      command
                                                  ),
                                                  src_party_id=job_initiator['party_id'],
                                                  dest_party_id=dest_party_id,
                                                  src_role=job_initiator['role'],
                                                  json_body=command_body if command_body else {},
                                                  work_mode=job.f_work_mode)
                    federated_response[dest_role][dest_party_id] = response
                except Exception as e:
                    federated_response[dest_role][dest_party_id] = {
                        "retcode": RetCode.FEDERATED_ERROR,
                        "retmsg": "Federated schedule error, {}".format(str(e))
                    }
                if federated_response[dest_role][dest_party_id]["retcode"]:
                    schedule_logger(job_id=job.f_job_id).error("An error occurred while {} the job to role {} party {}: \n{}".format(
                        command,
                        dest_role,
                        dest_party_id,
                        federated_response[dest_role][dest_party_id]["retmsg"]
                    ))
        return cls.return_federated_response(federated_response=federated_response)

    @classmethod
    def start_task(cls, job, task, task_parameters):
        return cls.task_command(job=job, task=task, command="start", command_body=task_parameters)

    @classmethod
    def stop_task(cls, job, task, stop_status):
        return cls.task_command(job=job, task=task, command="stop/{}".format(stop_status))

    @classmethod
    def task_command(cls, job, task, command, command_body=None):
        federated_response = {}
        roles, job_initiator = job.f_runtime_conf["role"], job.f_runtime_conf['initiator']
        for dest_role, dest_party_ids in roles.items():
            federated_response[dest_role] = {}
            for dest_party_id in dest_party_ids:
                try:
                    response = federated_api(job_id=task.f_job_id,
                                             method='POST',
                                             endpoint='/{}/schedule/{}/{}{}//{}/{}/{}/{}'.format(
                                                 API_VERSION,
                                                 task.f_job_id,
                                                 task.f_component_name,
                                                 task.f_task_id,
                                                 task.f_task_version,
                                                 task.f_role,
                                                 task.f_party_id,
                                                 command
                                             ),
                                             src_party_id=job_initiator['party_id'],
                                             dest_party_id=dest_party_id,
                                             src_role=job_initiator['role'],
                                             json_body=command_body if command_body else {},
                                             work_mode=RuntimeConfig.WORK_MODE)
                    federated_response[dest_role][dest_party_id] = response
                except Exception as e:
                    federated_response[dest_role][dest_party_id] = {
                        "retcode": RetCode.FEDERATED_ERROR,
                        "retmsg": "Federated schedule error, {}".format(str(e))
                    }
                if federated_response[dest_role][dest_party_id]["retcode"]:
                    schedule_logger(job_id=job.f_job_id).error("An error occurred while {} the task to role {} party {}: \n{}".format(
                        command,
                        dest_role,
                        dest_party_id,
                        federated_response[dest_role][dest_party_id]["retmsg"]
                    ))
        return cls.return_federated_response(federated_response=federated_response)

    @classmethod
    def report_task(cls, task: Task):
        """
        :param task:
        :return:
        """
        if task.f_role != task.f_initiator_role and task.f_party_id != task.f_initiator_party_id:
            # report
            task.f_run_ip = ""
            try:
                response = federated_api(job_id=task.f_job_id,
                                         method='POST',
                                         endpoint='/{}/schedule/{}/{}/{}/{}/{}/{}/status'.format(
                                             API_VERSION,
                                             task.f_job_id,
                                             task.f_component_name,
                                             task.f_task_id,
                                             task.f_task_version,
                                             task.f_role,
                                             task.f_party_id),
                                         src_party_id=task.f_party_id,
                                         dest_party_id=task.f_initiator_party_id,
                                         src_role=task.f_role,
                                         json_body=task.to_json(),
                                         work_mode=RuntimeConfig.WORK_MODE)
            except Exception as e:
                response = {
                    "retcode": RetCode.FEDERATED_ERROR,
                    "retmsg": "Federated error, {}".format(str(e))
                }
            if response["retcode"]:
                schedule_logger(job_id=task.f_job_id).error("An error occurred while {} the task to role {} party {}: \n{}".format(
                    "report",
                    task.f_initiator_role,
                    task.f_initiator_party_id,
                    response["retmsg"]
                ))
                return FederatedSchedulingStatusCode.FAILED
            else:
                return FederatedSchedulingStatusCode.SUCCESS
        else:
            return FederatedSchedulingStatusCode.SUCCESS

    @classmethod
    def return_federated_response(cls, federated_response):
        retcode_set = set()
        for dest_role in federated_response.keys():
            for party_id in federated_response[dest_role].keys():
                retcode_set.add(federated_response[dest_role][party_id]["retcode"])
        if len(retcode_set) == 1:
            if FederatedSchedulingStatusCode.SUCCESS in retcode_set:
                federated_scheduling_status_code = FederatedSchedulingStatusCode.SUCCESS
            else:
                federated_scheduling_status_code = FederatedSchedulingStatusCode.FAILED
        else:
            federated_scheduling_status_code = FederatedSchedulingStatusCode.PARTIAL
        return federated_scheduling_status_code, federated_response
