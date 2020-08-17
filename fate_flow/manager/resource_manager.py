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
from fate_arch.common import base_utils
from fate_flow.operation.job_saver import JobSaver
from fate_flow.utils import job_utils
from fate_flow.db.db_models import DB, ResourceRegistry, ResourceRecord
from fate_arch.common.conf_utils import get_base_config


class ResourceManager(object):
    @classmethod
    def initialize(cls):
        # initialize default
        # eggroll
        with DB.connection_context():
            resources = ResourceRegistry.select().where(ResourceRegistry.f_engine_id == "default_computing")
            if resources:
                resource = resources[0]
            else:
                resource = ResourceRegistry()
                resource.f_create_time = base_utils.current_timestamp()
                resource.f_remaining_cores = get_base_config("computing", {}).get("cores", None)
                resource.f_remaining_memory = get_base_config("computing", {}).get("memory", 0)
            resource.f_engine_id = "default_computing"
            resource.f_engine_type = get_base_config("computing", {}).get("engine", {})
            resource.f_engine_address = get_base_config("computing", {}).get("address", {})
            resource.f_cores = get_base_config("computing", {}).get("cores", None)
            resource.f_memory = get_base_config("computing", {}).get("memory", 0)
            resource.save()

    @classmethod
    def apply_for_resource_to_job(cls, job_id, role, party_id):
        status, engine_id, cores, memory = cls.resource_for_job(job_id=job_id, role=role, party_id=party_id, operation_type="apply")
        if status:
            try:
                with DB.connection_context():
                    resource_record = ResourceRecord()
                    resource_record.f_job_id = job_id
                    resource_record.f_role = role
                    resource_record.f_party_id = party_id
                    resource_record.f_engine_id = engine_id
                    resource_record.f_cores = cores
                    resource_record.f_memory = memory
                    resource_record.f_remaining_cores = cores
                    resource_record.f_remaining_memory = memory
                    resource_record.f_create_time = base_utils.current_timestamp()
                    rows = resource_record.save(force_insert=True)
                    if rows == 1:
                        schedule_logger(job_id=job_id).info(f"successfully apply resource for job {job_id} on {role} {party_id}")
                        return True
            except Exception as e:
                status, engine_id, cores, memory = cls.resource_for_job(job_id=job_id, role=role, party_id=party_id, operation_type="return")
                schedule_logger(job_id=job_id).warning(f"failed save resource record for job {job_id} on {role} {party_id}")
                return False
        else:
            schedule_logger(job_id=job_id).info(f"failed apply resource for job {job_id} on {role} {party_id}")
            return False

    @classmethod
    def return_resource(cls, job_id, role, party_id):
        status, engine_id, cores, memory = cls.resource_for_job(job_id=job_id, role=role, party_id=party_id, operation_type="return")
        if status:
            try:
                with DB.connection_context():
                    operate = ResourceRecord.delete().where(ResourceRecord.f_job_id==job_id, ResourceRecord.f_role==role, ResourceRecord.f_party_id==party_id)
                    if operate.execute() > 0:
                        schedule_logger(job_id=job_id).info(f"successfully return job {job_id} on {role} {party_id} resource")
            except Exception as e:
                schedule_logger(job_id=job_id).warning(f"failed delete job {job_id} on {role} {party_id} resource record")
            return True
        else:
            schedule_logger(job_id=job_id).info(f"failed return job {job_id} on {role} {party_id} resource")
            return False

    @classmethod
    def resource_for_job(cls, job_id, role, party_id, operation_type):
        dsl, runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=job_id,
                                                                                role=role,
                                                                                party_id=party_id)
        task_parallelism = runtime_conf["job_parameters"]["task_parallelism"]
        processors_per_task = runtime_conf["job_parameters"]["processors_per_task"]
        cores = task_parallelism * processors_per_task
        memory = 0

        with DB.connection_context():
            engine_id = "default_computing"
            engine_type = "EGGROLL"
            update_filters = [ResourceRegistry.f_engine_id == engine_id]
            if engine_type == "EGGROLL":
                cores_volume = cores if operation_type == "apply" else -cores
                memory_volume = 0
            else:
                cores_volume = 0
                memory_volume = 0
            if operation_type == "apply":
                update_filters.append(ResourceRegistry.f_remaining_cores >= cores_volume)
                update_filters.append(ResourceRegistry.f_remaining_memory >= memory_volume)
                operate = ResourceRegistry.update({ResourceRegistry.f_remaining_cores: ResourceRegistry.f_remaining_cores - cores_volume,
                                                   ResourceRegistry.f_remaining_memory: ResourceRegistry.f_remaining_memory - memory_volume}
                                                  ).where(*update_filters)
            elif operation_type == "return":
                operate = ResourceRegistry.update({ResourceRegistry.f_remaining_cores: ResourceRegistry.f_remaining_cores + cores_volume,
                                                   ResourceRegistry.f_remaining_memory: ResourceRegistry.f_remaining_memory + memory_volume}
                                                  ).where(*update_filters)
            else:
                raise RuntimeError(f"can not support {operation_type} operation type")
            return operate.execute() > 0, engine_id, cores, memory

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
