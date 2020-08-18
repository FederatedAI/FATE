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

from fate_arch.common import base_utils
from fate_arch.common.conf_utils import get_base_config
from fate_arch.common.log import schedule_logger
from fate_flow.db.db_models import DB, ResourceRegistry, Job
from fate_flow.entity.constant import ResourceOperation
from fate_flow.settings import stat_logger
from fate_flow.utils import job_utils


class ResourceManager(object):
    @classmethod
    def initialize(cls):
        # initialize default
        with DB.connection_context():
            for engine_id, engine_info in get_base_config("computing", {}).items():
                resources = ResourceRegistry.select().where(ResourceRegistry.f_engine_id == engine_id)
                is_insert = False
                if resources:
                    resource = resources[0]
                else:
                    resource = ResourceRegistry()
                    resource.f_create_time = base_utils.current_timestamp()
                    resource.f_remaining_cores = engine_info.get("cores", 0)
                    resource.f_remaining_memory = engine_info.get("memory", 0)
                    is_insert = True
                resource.f_engine_id = engine_id
                resource.f_engine_type = engine_info.get("engine", {})
                resource.f_engine_address = engine_info.get("address", {})
                resource.f_cores = engine_info.get("cores", 0)
                resource.f_memory = engine_info.get("memory", 0)
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
        engine_id, cores, memory = cls.calculate_job_resource(job_id=job_id, role=role, party_id=party_id)
        apply_status = cls.update_resource(model=ResourceRegistry,
                                           cores=cores,
                                           memory=memory,
                                           operation_type=ResourceOperation.APPLY,
                                           engine_id=engine_id,
                                           )
        if apply_status:
            update_fields = {
                Job.f_engine_id: engine_id,
                Job.f_cores: cores,
                Job.f_memory: memory,
                Job.f_remaining_cores: cores,
                Job.f_remaining_memory: memory,
            }
            filter_fields = [
                Job.f_job_id == job_id,
                Job.f_role == role,
                Job.f_party_id == party_id,
                Job.f_cores == 0,
                Job.f_memory == 0
            ]
            with DB.connection_context():
                operate = Job.update(update_fields).where(*filter_fields)
                update_status = operate.execute() > 0
            if update_status:
                schedule_logger(job_id=job_id).info(f"apply job {job_id} resource on {role} {party_id} successfully")
                return True
            else:
                schedule_logger(job_id=job_id).info(
                    f"save apply job {job_id} resource on {role} {party_id} record failed, rollback...")
                cls.return_job_resource(job_id=job_id, role=role, party_id=party_id)
                return False
        else:
            schedule_logger(job_id=job_id).info(f"apply job {job_id} resource on {role} {party_id} failed")
            return False

    @classmethod
    def return_job_resource(cls, job_id, role, party_id):
        engine_id, cores, memory = cls.calculate_job_resource(job_id=job_id, role=role, party_id=party_id)
        update_fields = {
            Job.f_engine_id: engine_id,
            Job.f_cores: 0,
            Job.f_memory: 0,
            Job.f_remaining_cores: 0,
            Job.f_remaining_memory: 0,
        }
        filter_fields = [
            Job.f_job_id == job_id,
            Job.f_role == role,
            Job.f_party_id == party_id,
            Job.f_cores == cores,
            Job.f_memory == memory
        ]
        with DB.connection_context():
            operate = Job.update(update_fields).where(*filter_fields)
            update_status = operate.execute() > 0
        if update_status:
            return_status = cls.update_resource(model=ResourceRegistry,
                                                cores=cores,
                                                memory=memory,
                                                operation_type=ResourceOperation.RETURN,
                                                engine_id=engine_id,
                                                )
            if return_status:
                schedule_logger(job_id=job_id).info(f"return job {job_id} resource on {role} {party_id} successfully")
                return True
            else:
                schedule_logger(job_id=job_id).info(f"return job {job_id} resource on {role} {party_id} failed")
                return False
        else:
            schedule_logger(job_id=job_id).info(f"save return job {job_id} resource on {role} {party_id} record failed")
            return False

    @classmethod
    def calculate_job_resource(cls, job_id, role, party_id):
        dsl, runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=job_id,
                                                                                role=role,
                                                                                party_id=party_id)
        task_parallelism = runtime_conf["job_parameters"]["task_parallelism"]
        cores_per_task = runtime_conf["job_parameters"]["cores_per_task"]
        memory_per_task = runtime_conf["job_parameters"]["memory_per_task"]
        cores = task_parallelism * cores_per_task
        memory = task_parallelism * memory_per_task
        engine_id = runtime_conf["job_parameters"]["computing_backend"]
        return engine_id, cores, memory

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

        update_status = cls.update_resource(model=Job,
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
