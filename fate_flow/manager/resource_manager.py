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

import operator
from fate_arch.common.log import schedule_logger
from fate_arch.common import base_utils
from fate_flow.utils import job_utils
from fate_flow.db.db_models import DB, ResourceRegistry, ResourceRecord
from fate_arch.common.conf_utils import get_base_config
from fate_flow.settings import stat_logger
from fate_flow.entity.constant import ResourceOperation


class ResourceManager(object):
    @classmethod
    def initialize(cls):
        # initialize default
        # eggroll
        with DB.connection_context():
            resources = ResourceRegistry.select().where(ResourceRegistry.f_engine_id == "default_computing")
            is_insert = False
            if resources:
                resource = resources[0]
            else:
                resource = ResourceRegistry()
                resource.f_create_time = base_utils.current_timestamp()
                resource.f_remaining_cores = get_base_config("computing", {}).get("cores", None)
                resource.f_remaining_memory = get_base_config("computing", {}).get("memory", 0)
                is_insert = True
            resource.f_engine_id = "default_computing"
            resource.f_engine_type = get_base_config("computing", {}).get("engine", {})
            resource.f_engine_address = get_base_config("computing", {}).get("address", {})
            resource.f_cores = get_base_config("computing", {}).get("cores", None)
            resource.f_memory = get_base_config("computing", {}).get("memory", 0)
            if is_insert:
                try:
                    resource.save(force_insert=True)
                except Exception as e:
                    stat_logger.warning(e)
                stat_logger.info(f"initialize default computing engine")
            else:
                resource.save()
                stat_logger.info(f"update default computing engine")

    @classmethod
    def apply_for_job_resource(cls, job_id, role, party_id):
        status, engine_id, cores, memory = cls.resource_for_job(job_id=job_id, role=role, party_id=party_id, operation_type=ResourceOperation.APPLY)
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
                    resource_record.f_in_use = True
                    resource_record.f_create_time = base_utils.current_timestamp()
                    rows = resource_record.save(force_insert=True)
                    if rows == 1:
                        schedule_logger(job_id=job_id).info(f"successfully apply resource for job {job_id} on {role} {party_id}")
                        return True
            except Exception as e:
                status, engine_id, cores, memory = cls.resource_for_job(job_id=job_id, role=role, party_id=party_id, operation_type=ResourceOperation.RETURN)
                schedule_logger(job_id=job_id).warning(f"failed save resource record for job {job_id} on {role} {party_id}")
                return False
        else:
            schedule_logger(job_id=job_id).info(f"failed apply resource for job {job_id} on {role} {party_id}")
            return False

    @classmethod
    def return_job_resource(cls, job_id, role, party_id):
        status, engine_id, cores, memory = cls.resource_for_job(job_id=job_id, role=role, party_id=party_id, operation_type=ResourceOperation.RETURN)
        if status:
            try:
                with DB.connection_context():
                    job_resource_records = ResourceRecord.select().where(ResourceRecord.f_job_id==job_id, ResourceRecord.f_role==role, ResourceRecord.f_party_id==party_id)
                    if not job_resource_records:
                        schedule_logger(job_id=job_id).warning(f"can not found job {job_id} {role} {party_id} resource record")
                    job_resource_record = job_resource_records[0]
                    job_resource_record.f_in_use = False
                    operate = ResourceRecord.update({ResourceRecord.f_in_use: False}).where(ResourceRecord.f_job_id==job_id, ResourceRecord.f_role==role, ResourceRecord.f_party_id==party_id)
                    if operate.execute() > 0:
                        schedule_logger(job_id=job_id).info(f"successfully return job {job_id} on {role} {party_id} resource and invalidate the record")
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
        cores_per_task = runtime_conf["job_parameters"]["cores_per_task"]
        memory_per_task = runtime_conf["job_parameters"]["memory_per_task"]
        cores = task_parallelism * cores_per_task
        memory = task_parallelism * memory_per_task

        engine_id = "default_computing"
        engine_type = "EGGROLL"
        if engine_type == "EGGROLL":
            cores_on_engine = cores
            memory_on_engine = 0
        else:
            cores_on_engine = 0
            memory_on_engine = 0
        schedule_logger(job_id=job_id).info(
            "try {} job {} resource on {} {}".format(operation_type, job_id, role, party_id))
        update_status = cls.update_resource(model=ResourceRegistry,
                                            cores=cores_on_engine,
                                            memory=memory_on_engine,
                                            operation_type=operation_type,
                                            engine_id=engine_id,
                                            )
        if update_status:
            schedule_logger(job_id=job_id).info(
                "{} job {} resource on {} {} successfully".format(operation_type, job_id, role, party_id))
        else:
            schedule_logger(job_id=job_id).info(
                "{} job {} resource on {} {} failed".format(operation_type, job_id, role, party_id))
        return update_status, engine_id, cores, memory

    @classmethod
    def apply_for_task_resource(cls, task_info):
        return ResourceManager.resource_for_task(task_info=task_info, operation_type=ResourceOperation.APPLY)

    @classmethod
    def return_task_resource(cls, task_info):
        return ResourceManager.resource_for_task(task_info=task_info, operation_type=ResourceOperation.RETURN)

    @classmethod
    def resource_for_task(cls, task_info, operation_type):
        dsl, runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=task_info["job_id"],
                                                                                role=task_info["role"],
                                                                                party_id=task_info["party_id"])
        cores_per_task = runtime_conf["job_parameters"]["cores_per_task"]
        memory_per_task = runtime_conf["job_parameters"]["memory_per_task"]
        schedule_logger(job_id=task_info["job_id"]).info(
            "try {} job {} resource to task {} {}".format(operation_type, task_info["job_id"], task_info["task_id"],
                                                          task_info["task_version"]))

        update_status = cls.update_resource(model=ResourceRecord,
                                            cores=cores_per_task,
                                            memory=memory_per_task,
                                            operation_type=operation_type,
                                            job_id=task_info["job_id"],
                                            role=task_info["role"],
                                            party_id=task_info["party_id"],
                                            )
        if update_status:
            schedule_logger(job_id=task_info["job_id"]).info(
                "{} job {} resource to task {} {} successfully".format(operation_type, task_info["job_id"],
                                                                       task_info["task_id"], task_info["task_version"]))
        else:
            schedule_logger(job_id=task_info["job_id"]).info(
                "{} job {} resource to task {} {} failed".format(operation_type, task_info["job_id"],
                                                                 task_info["task_id"], task_info["task_version"]))
        return update_status

    @classmethod
    def update_resource(cls, model, cores, memory, operation_type, **kwargs):
        filters = []
        for p_k in model.get_primary_keys_name():
            filters.append(operator.attrgetter(p_k)(model) == kwargs[p_k.lstrip("f_")])
        with DB.connection_context():
            if operation_type == ResourceOperation.APPLY:
                filters.append(model.f_remaining_cores >= cores)
                filters.append(model.f_remaining_memory >= memory)
                operate = model.update({model.f_remaining_cores: model.f_remaining_cores - cores,
                                        model.f_remaining_memory: model.f_remaining_memory - memory}
                                       ).where(*filters)
            elif operation_type == ResourceOperation.RETURN:
                operate = model.update({model.f_remaining_cores: model.f_remaining_cores + cores,
                                        model.f_remaining_memory: model.f_remaining_memory + memory}
                                       ).where(*filters)
            else:
                raise RuntimeError(f"can not support {operation_type} resource operation type")
            return operate.execute() > 0
