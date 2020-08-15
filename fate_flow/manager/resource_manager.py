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

from fate_arch.common.log import schedule_logger
from fate_flow.operation.job_saver import JobSaver
from fate_flow.utils import job_utils


class ResourceManager(object):
    @classmethod
    def apply_for_resource_to_job(cls, job_id, role, party_id):
        return cls.resource_for_job(job_id=job_id, role=role, party_id=party_id, operation_type="apply")

    @classmethod
    def return_resource(cls, job_id, role, party_id):
        pass

    @classmethod
    def resource_for_job(cls, job_id, role, party_id, operation_type):
        dsl, runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=job_id,
                                                                                role=role,
                                                                                party_id=party_id)
        task_parallelism = runtime_conf["job_parameters"]["task_parallelism"]
        processors_per_task = runtime_conf["job_parameters"]["processors_per_task"]
        cores = task_parallelism * processors_per_task
        return True, cores

    @classmethod
    def apply_for_resource_to_task(cls, task_info):
        return ResourceManager.resource_for_task(task_info=task_info, operation_type="apply")

    @classmethod
    def return_resource_to_job(cls, task_info):
        return ResourceManager.resource_for_task(task_info=task_info, operation_type="return")

    @classmethod
    def resource_for_task(cls, task_info, operation_type):
        dsl, runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=task_info["job_id"],
                                                                                role=task_info["role"],
                                                                                party_id=task_info["party_id"])
        processors_per_task = runtime_conf["job_parameters"]["processors_per_task"]
        schedule_logger(job_id=task_info["job_id"]).info(
            "Try {} job {} resource to task {} {}".format(operation_type, task_info["job_id"], task_info["task_id"],
                                                          task_info["task_version"]))
        update_status = JobSaver.update_job_resource(job_id=task_info["job_id"], role=task_info["role"],
                                                     party_id=task_info["party_id"], volume=(
                processors_per_task if operation_type == "apply" else -processors_per_task))
        if update_status:
            schedule_logger(job_id=task_info["job_id"]).info(
                "Successfully {} job {} resource to task {} {}".format(operation_type, task_info["job_id"],
                                                                       task_info["task_id"], task_info["task_version"]))
        else:
            schedule_logger(job_id=task_info["job_id"]).info(
                "Failed {} job {} resource to task {} {}".format(operation_type, task_info["job_id"],
                                                                 task_info["task_id"], task_info["task_version"]))
        return update_status
