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
from fate_flow.utils import job_utils


class FederatedScheduler(object):
    """
    Send commands to party,
    Report info to initiator
    """

    # Job
    @classmethod
    def create_job(cls, job: Job):
        retcode, response = cls.job_command(job=job, command="create", command_body=job.to_dict_info())
        if retcode != 0:
            raise Exception("Create job failed: {}".format(json_dumps(response, indent=4)))

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
    def sync_job(cls, job, update_fields):
        schedule_logger(job_id=job.f_job_id).info("Job {} is {}, sync to all party".format(job.f_job_id, job.f_status))
        status_code, response = cls.job_command(job=job, command="update", command_body=job.to_dict_info(only_primary_with=update_fields))
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("Sync job {} status {} to all party success".format(job.f_job_id, job.f_status))
        else:
            schedule_logger(job_id=job.f_job_id).info("Sync job {} status {} to all party failed: \n{}".format(job.f_job_id, job.f_status, response))
        return status_code, response

    @classmethod
    def save_pipelined_model(cls, job):
        schedule_logger(job_id=job.f_job_id).info("Try to save job {} pipelined model".format(job.f_job_id))
        status_code, response = cls.job_command(job=job, command="model")
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("Save job {} pipelined model success".format(job.f_job_id))
        else:
            schedule_logger(job_id=job.f_job_id).info("Save job {} pipelined model failed:\n{}".format(job.f_job_id, response))
        return status_code, response

    @classmethod
    def stop_job(cls, job, stop_status):
        schedule_logger(job_id=job.f_job_id).info("Try to stop job {}".format(job.f_job_id))
        job.f_status = stop_status
        status_code, response = cls.job_command(job=job, command="stop/{}".format(stop_status))
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("Stop job {} success".format(job.f_job_id))
        else:
            schedule_logger(job_id=job.f_job_id).info("Stop job {} failed:\n{}".format(job.f_job_id, response))
        return status_code, response

    @classmethod
    def request_stop_job(cls, job, stop_status):
        return cls.job_command(job=job, command="stop/{}".format(stop_status), dest_only_initiator=True)

    @classmethod
    def cancel_job(cls, job):
        return cls.job_command(job=job, command="cancel")

    @classmethod
    def clean_job(cls, job):
        schedule_logger(job_id=job.f_job_id).info("Try to clean job {}".format(job.f_job_id))
        status_code, response = cls.job_command(job=job, command="clean", command_body=job.f_runtime_conf["role"].copy())
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("Clean job {} success".format(job.f_job_id))
        else:
            schedule_logger(job_id=job.f_job_id).info("Clean job {} failed:\n{}".format(job.f_job_id, response))
        return status_code, response

    @classmethod
    def job_command(cls, job, command, command_body=None, dest_only_initiator=False):
        federated_response = {}
        roles, job_initiator = job.f_runtime_conf["role"], job.f_runtime_conf['initiator']
        if not dest_only_initiator:
            dest_partys = roles.items()
            api_type = "control"
        else:
            dest_partys = [(job_initiator["role"], [job_initiator["party_id"]])]
            api_type = "initiator"
        for dest_role, dest_party_ids in dest_partys:
            federated_response[dest_role] = {}
            for dest_party_id in dest_party_ids:
                try:
                    response = federated_api(job_id=job.f_job_id,
                                                  method='POST',
                                                  endpoint='/{}/{}/{}/{}/{}/{}'.format(
                                                      API_VERSION,
                                                      api_type,
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

    # TaskSet
    @classmethod
    def sync_task_set(cls, job, task_set, update_fields):
        schedule_logger(job_id=task_set.f_job_id).info("Job {} task set {} is {}, sync to all party".format(task_set.f_job_id, task_set.f_task_set_id, task_set.f_status))
        status_code, response = cls.task_set_command(job=job, task_set=task_set, command="update", command_body=task_set.to_dict_info(only_primary_with=update_fields))
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=task_set.f_job_id).info("Sync job {} task set {} status {} to all party success".format(task_set.f_job_id, task_set.f_task_set_id, task_set.f_status))
        else:
            schedule_logger(job_id=task_set.f_job_id).info("Sync job {} task set {} status {} to all party failed: \n{}".format(task_set.f_job_id, task_set.f_task_set_id, task_set.f_status, response))
        return status_code, response

    @classmethod
    def stop_task_set(cls, job, task_set, stop_status):
        schedule_logger(job_id=task_set.f_job_id).info("Try to stop job {} task set {}".format(task_set.f_job_id, task_set.f_task_set_id))
        task_set.f_status = stop_status
        status_code, response = cls.task_set_command(job=job, task_set=task_set, command="stop/{}".format(stop_status))
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("Stop job {} task set {} success".format(task_set.f_job_id, task_set.f_task_set_id))
        else:
            schedule_logger(job_id=job.f_job_id).info("Stop job {} task set {} failed:\n{}".format(task_set.f_job_id, task_set.f_task_set_id, response))
        return status_code, response

    @classmethod
    def task_set_command(cls, job, task_set, command, command_body=None):
        federated_response = {}
        roles, job_initiator = job.f_runtime_conf["role"], job.f_runtime_conf['initiator']
        for dest_role, dest_party_ids in roles.items():
            federated_response[dest_role] = {}
            for dest_party_id in dest_party_ids:
                try:
                    response = federated_api(job_id=job.f_job_id,
                                             method='POST',
                                             endpoint='/{}/control/{}/{}/{}/{}/{}'.format(
                                                 API_VERSION,
                                                 job.f_job_id,
                                                 task_set.f_task_set_id,
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
                    schedule_logger(job_id=job.f_job_id).error("An error occurred while {} the task set to role {} party {}: \n{}".format(
                        command,
                        dest_role,
                        dest_party_id,
                        federated_response[dest_role][dest_party_id]["retmsg"]
                    ))
        return cls.return_federated_response(federated_response=federated_response)

    # Task
    @classmethod
    def start_task(cls, job, task, task_parameters):
        return cls.task_command(job=job, task=task, command="start", command_body=task_parameters)

    @classmethod
    def sync_task(cls, job, task, update_fields):
        schedule_logger(job_id=task.f_job_id).info("Job {} task {} {} is {}, sync to all party".format(task.f_job_id, task.f_task_id, task.f_task_version, task.f_status))
        status_code, response = cls.task_command(job=job, task=task, command="update", command_body=task.to_dict_info(only_primary_with=update_fields))
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=task.f_job_id).info("Sync job {} task {} {} status {} to all party success".format(task.f_job_id, task.f_task_id, task.f_task_version, task.f_status))
        else:
            schedule_logger(job_id=task.f_job_id).info("Sync job {} task {} {} status {} to all party failed: \n{}".format(task.f_job_id, task.f_task_id, task.f_task_version, task.f_status, response))
        return status_code, response

    @classmethod
    def stop_task(cls, job, task, stop_status):
        schedule_logger(job_id=task.f_job_id).info("Try to stop job {} task {} {}".format(task.f_job_id, task.f_task_id, task.f_task_version))
        task.f_status = stop_status
        status_code, response = cls.task_command(job=job, task=task, command="stop/{}".format(stop_status))
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            schedule_logger(job_id=job.f_job_id).info("Stop job {} task {} {} success".format(task.f_job_id, task.f_task_id, task.f_task_version))
        else:
            schedule_logger(job_id=job.f_job_id).info("Stop job {} task {} {} failed:\n{}".format(task.f_job_id, task.f_task_id, task.f_task_version, response))
        return status_code, response

    @classmethod
    def task_command(cls, job, task, command, command_body=None):
        federated_response = {}
        roles, job_initiator = job.f_runtime_conf["role"], job.f_runtime_conf['initiator']
        dsl_parser = job_utils.get_job_dsl_parser(dsl=job.f_dsl, runtime_conf=job.f_runtime_conf, train_runtime_conf=job.f_train_runtime_conf)
        component = dsl_parser.get_component_info(component_name=task.f_component_name)
        component_parameters = component.get_role_parameters()
        for dest_role, parameters_on_partys in component_parameters.items():
            federated_response[dest_role] = {}
            for parameters_on_party in parameters_on_partys:
                dest_party_id = parameters_on_party.get('local', {}).get('party_id')
                try:
                    response = federated_api(job_id=task.f_job_id,
                                             method='POST',
                                             endpoint='/{}/control/{}/{}/{}/{}/{}/{}/{}'.format(
                                                 API_VERSION,
                                                 task.f_job_id,
                                                 task.f_component_name,
                                                 task.f_task_id,
                                                 task.f_task_version,
                                                 dest_role,
                                                 dest_party_id,
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
    def report_task_to_initiator(cls, task: Task):
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
                                         endpoint='/{}/control/{}/{}/{}/{}/{}/{}/status'.format(
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
                                         json_body=task.to_dict_info(),
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

    # Utils
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
