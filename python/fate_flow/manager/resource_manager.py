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

import math
import typing

from fate_arch.common import EngineType
from fate_arch.common import base_utils
from fate_arch.common.conf_utils import get_base_config
from fate_arch.common.log import schedule_logger
from fate_arch.computing import ComputingEngine
from fate_flow.db.db_models import DB, EngineRegistry, Job
from fate_flow.entity.types import ResourceOperation, RunParameters
from fate_flow.settings import stat_logger, STANDALONE_BACKEND_VIRTUAL_CORES_PER_NODE, SUPPORT_BACKENDS_ENTRANCE, \
    MAX_CORES_PERCENT_PER_JOB, DEFAULT_TASK_CORES_PER_NODE
from fate_flow.utils import job_utils


class ResourceManager(object):
    @classmethod
    def initialize(cls):
        for backend_name, backend_engines in SUPPORT_BACKENDS_ENTRANCE.items():
            for engine_type, engine_keys in backend_engines.items():
                engine_config = get_base_config(backend_name, {}).get(engine_keys[1], {})
                if engine_config:
                    cls.register_engine(engine_type=engine_type, engine_name=engine_keys[0], engine_entrance=engine_keys[1], engine_config=engine_config)

        # initialize standalone engine
        for backend_engines in SUPPORT_BACKENDS_ENTRANCE.values():
            for engine_type in backend_engines.keys():
                engine_name = "STANDALONE"
                engine_entrance = "fateflow"
                engine_config = {
                    "nodes": 1,
                    "cores_per_node": STANDALONE_BACKEND_VIRTUAL_CORES_PER_NODE,
                }
                cls.register_engine(engine_type=engine_type, engine_name=engine_name, engine_entrance=engine_entrance, engine_config=engine_config)

    @classmethod
    @DB.connection_context()
    def register_engine(cls, engine_type, engine_name, engine_entrance, engine_config):
        nodes = engine_config.get("nodes", 1)
        cores = engine_config.get("cores_per_node", 0) * nodes
        memory = engine_config.get("memory_per_node", 0) * nodes
        filters = [EngineRegistry.f_engine_type == engine_type, EngineRegistry.f_engine_name == engine_name]
        resources = EngineRegistry.select().where(*filters)
        if resources:
            resource = resources[0]
            update_fields = {}
            update_fields[EngineRegistry.f_engine_config] = engine_config
            update_fields[EngineRegistry.f_cores] = cores
            update_fields[EngineRegistry.f_memory] = memory
            update_fields[EngineRegistry.f_remaining_cores] = EngineRegistry.f_remaining_cores + (
                    cores - resource.f_cores)
            update_fields[EngineRegistry.f_remaining_memory] = EngineRegistry.f_remaining_memory + (
                    memory - resource.f_memory)
            update_fields[EngineRegistry.f_nodes] = nodes
            operate = EngineRegistry.update(update_fields).where(*filters)
            update_status = operate.execute() > 0
            if update_status:
                stat_logger.info(f"update {engine_type} engine {engine_name} {engine_entrance} registration information")
            else:
                stat_logger.info(f"update {engine_type} engine {engine_name} {engine_entrance} registration information takes no effect")
        else:
            resource = EngineRegistry()
            resource.f_create_time = base_utils.current_timestamp()
            resource.f_engine_type = engine_type
            resource.f_engine_name = engine_name
            resource.f_engine_entrance = engine_entrance
            resource.f_engine_config = engine_config

            resource.f_cores = cores
            resource.f_memory = memory
            resource.f_remaining_cores = cores
            resource.f_remaining_memory = memory
            resource.f_nodes = nodes
            try:
                resource.save(force_insert=True)
            except Exception as e:
                stat_logger.warning(e)
            stat_logger.info(f"create {engine_type} engine {engine_name} {engine_entrance} registration information")

    @classmethod
    def check_resource_apply(cls, job_parameters: RunParameters, engines_info):
        computing_engine, cores, memory = cls.calculate_job_resource(job_parameters=job_parameters)
        max_cores_per_job = math.floor(engines_info[EngineType.COMPUTING].f_cores * MAX_CORES_PERCENT_PER_JOB)
        if cores > max_cores_per_job:
            return False, max_cores_per_job
        else:
            return True, max_cores_per_job

    @classmethod
    def apply_for_job_resource(cls, job_id, role, party_id):
        return cls.resource_for_job(job_id=job_id, role=role, party_id=party_id, operation_type=ResourceOperation.APPLY)

    @classmethod
    def return_job_resource(cls, job_id, role, party_id):
        return cls.resource_for_job(job_id=job_id, role=role, party_id=party_id,
                                    operation_type=ResourceOperation.RETURN)

    @classmethod
    @DB.connection_context()
    def resource_for_job(cls, job_id, role, party_id, operation_type):
        operate_status = False
        engine_name, cores, memory = cls.calculate_job_resource(job_id=job_id, role=role, party_id=party_id)
        try:
            with DB.atomic():
                updates = {
                    Job.f_engine_type: EngineType.COMPUTING,
                    Job.f_engine_name: engine_name,
                    Job.f_cores: cores,
                    Job.f_memory: memory,
                }
                filters = [
                    Job.f_job_id == job_id,
                    Job.f_role == role,
                    Job.f_party_id == party_id,
                ]
                if operation_type == ResourceOperation.APPLY:
                    updates[Job.f_remaining_cores] = cores
                    updates[Job.f_remaining_memory] = memory
                    updates[Job.f_resource_in_use] = True
                    updates[Job.f_apply_resource_time] = base_utils.current_timestamp()
                    filters.append(Job.f_resource_in_use == False)
                elif operation_type == ResourceOperation.RETURN:
                    updates[Job.f_resource_in_use] = False
                    updates[Job.f_return_resource_time] = base_utils.current_timestamp()
                    filters.append(Job.f_resource_in_use == True)
                operate = Job.update(updates).where(*filters)
                record_status = operate.execute() > 0
                if not record_status:
                    raise RuntimeError(f"record job {job_id} resource {operation_type} failed on {role} {party_id}")

                filters, updates = cls.update_resource_sql(resource_model=EngineRegistry,
                                                           cores=cores,
                                                           memory=memory,
                                                           operation_type=operation_type,
                                                           )
                filters.append(EngineRegistry.f_engine_type == EngineType.COMPUTING)
                filters.append(EngineRegistry.f_engine_name == engine_name)
                operate = EngineRegistry.update(updates).where(*filters)
                apply_status = operate.execute() > 0
                if not apply_status:
                    raise RuntimeError(
                        f"{operation_type} resource from engine {engine_name} for job {job_id} resource {operation_type} failed on {role} {party_id}")
            operate_status = True
        except Exception as e:
            schedule_logger(job_id=job_id).warning(e)
            schedule_logger(job_id=job_id).warning(
                f"{operation_type} job {job_id} resource(cores {cores} memory {memory}) on {role} {party_id} failed")
            operate_status = False
        finally:
            remaining_cores, remaining_memory = cls.get_remaining_resource(EngineRegistry,
                                                                           [
                                                                               EngineRegistry.f_engine_type == EngineType.COMPUTING,
                                                                               EngineRegistry.f_engine_name == engine_name])
            operate_msg = "successfully" if operate_status else "failed"
            schedule_logger(job_id=job_id).info(
                f"{operation_type} job {job_id} resource(cores {cores} memory {memory}) on {role} {party_id} {operate_msg}, remaining cores: {remaining_cores} remaining memory: {remaining_memory}")
            return operate_status

    @classmethod
    def adapt_engine_parameters(cls, role, job_parameters: RunParameters, create_initiator_baseline=False):
        computing_engine_info = ResourceManager.get_engine_registration_info(engine_type=EngineType.COMPUTING,
                                                                             engine_name=job_parameters.computing_engine)
        if create_initiator_baseline:
            job_parameters.adaptation_parameters = {
                "task_nodes": 0,
                "task_cores_per_node": 0,
                "task_memory_per_node": 0,
            }
            task_cores = 0
        else:
            # use initiator baseline
            if role == "arbiter":
                task_cores = 1
            else:
                task_cores = job_parameters.adaptation_parameters["task_nodes"] * job_parameters.adaptation_parameters["task_cores_per_node"]

        if job_parameters.computing_engine in {ComputingEngine.STANDALONE, ComputingEngine.EGGROLL}:
            job_parameters.adaptation_parameters["task_nodes"] = computing_engine_info.f_nodes
            job_parameters.adaptation_parameters["task_cores_per_node"] = int(
                job_parameters.eggroll_run.get("eggroll.session.processors.per.node",
                                               cls.adapt_task_cores_per_node(create_initiator_baseline, task_cores, job_parameters.adaptation_parameters["task_nodes"])
                                               )
            )
            if not create_initiator_baseline:
                job_parameters.eggroll_run["eggroll.session.processors.per.node"] = job_parameters.adaptation_parameters["task_cores_per_node"]
        elif job_parameters.computing_engine == ComputingEngine.SPARK:
            job_parameters.adaptation_parameters["task_nodes"] = int(job_parameters.spark_run.get("num-executors", computing_engine_info.f_nodes))
            job_parameters.adaptation_parameters["task_cores_per_node"] = int(
                job_parameters.spark_run.get("executor-cores",
                                             cls.adapt_task_cores_per_node(create_initiator_baseline, task_cores, job_parameters.adaptation_parameters["task_nodes"])
                                             )
            )
            if not create_initiator_baseline:
                job_parameters.spark_run["num-executors"] = job_parameters.adaptation_parameters["task_nodes"]
                job_parameters.spark_run["executor-cores"] = job_parameters.adaptation_parameters["task_cores_per_node"]

    @classmethod
    def adapt_task_cores_per_node(cls, create_initiator_baseline, initiator_baseline, task_nodes):
        if create_initiator_baseline:
            return DEFAULT_TASK_CORES_PER_NODE
        else:
            return max(1, int(initiator_baseline / task_nodes))

    @classmethod
    def calculate_job_resource(cls, job_parameters: RunParameters = None, job_id=None, role=None, party_id=None):
        if not job_parameters:
            dsl, runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=job_id,
                                                                                    role=role,
                                                                                    party_id=party_id)
            job_parameters = RunParameters(**runtime_conf["job_parameters"])
        cores = job_parameters.adaptation_parameters["task_cores_per_node"] * job_parameters.adaptation_parameters[
            "task_nodes"] * job_parameters.task_parallelism
        memory = job_parameters.adaptation_parameters["task_memory_per_node"] * job_parameters.adaptation_parameters[
            "task_nodes"] * job_parameters.task_parallelism
        return job_parameters.computing_engine, cores, memory

    @classmethod
    def calculate_task_resource(cls, task_parameters: RunParameters = None, task_info: dict = None):
        if not task_parameters:
            dsl, runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=task_info["job_id"],
                                                                                    role=task_info["role"],
                                                                                    party_id=task_info["party_id"])
            task_parameters = RunParameters(**runtime_conf["job_parameters"])
        cores_per_task = task_parameters.adaptation_parameters["task_cores_per_node"] * \
                         task_parameters.adaptation_parameters["task_nodes"]
        memory_per_task = task_parameters.adaptation_parameters["task_memory_per_node"] * \
                          task_parameters.adaptation_parameters["task_nodes"]
        return cores_per_task, memory_per_task

    @classmethod
    def apply_for_task_resource(cls, task_info):
        return ResourceManager.resource_for_task(task_info=task_info, operation_type=ResourceOperation.APPLY)

    @classmethod
    def return_task_resource(cls, task_info):
        return ResourceManager.resource_for_task(task_info=task_info, operation_type=ResourceOperation.RETURN)

    @classmethod
    def resource_for_task(cls, task_info, operation_type):
        cores_per_task, memory_per_task = cls.calculate_task_resource(task_info=task_info)

        filters, updates = cls.update_resource_sql(resource_model=Job,
                                                   cores=cores_per_task,
                                                   memory=memory_per_task,
                                                   operation_type=operation_type,
                                                   )
        filters.append(Job.f_job_id == task_info["job_id"])
        filters.append(Job.f_role == task_info["role"])
        filters.append(Job.f_party_id == task_info["party_id"])
        operate = Job.update(updates).where(*filters)
        operate_status = operate.execute() > 0
        if operate_status:
            schedule_logger(job_id=task_info["job_id"]).info(
                "task {} {} {} resource successfully".format(task_info["task_id"],
                                                             task_info["task_version"], operation_type))
        else:
            schedule_logger(job_id=task_info["job_id"]).warning(
                "task {} {} {} resource failed".format(task_info["task_id"],
                                                       task_info["task_version"], operation_type))
        return operate_status

    @classmethod
    def update_resource_sql(cls, resource_model: typing.Union[EngineRegistry, Job], cores, memory, operation_type):
        if operation_type == ResourceOperation.APPLY:
            filters = [
                resource_model.f_remaining_cores >= cores,
                resource_model.f_remaining_memory >= memory
            ]
            updates = {resource_model.f_remaining_cores: resource_model.f_remaining_cores - cores,
                       resource_model.f_remaining_memory: resource_model.f_remaining_memory - memory}
        elif operation_type == ResourceOperation.RETURN:
            filters = []
            updates = {resource_model.f_remaining_cores: resource_model.f_remaining_cores + cores,
                       resource_model.f_remaining_memory: resource_model.f_remaining_memory + memory}
        else:
            raise RuntimeError(f"can not support {operation_type} resource operation type")
        return filters, updates

    @classmethod
    @DB.connection_context()
    def get_remaining_resource(cls, resource_model: typing.Union[EngineRegistry, Job], filters):
        remaining_cores, remaining_memory = None, None
        try:
            objs = resource_model.select(resource_model.f_remaining_cores, resource_model.f_remaining_memory).where(
                *filters)
            if objs:
                remaining_cores, remaining_memory = objs[0].f_remaining_cores, objs[0].f_remaining_memory
        except Exception as e:
            schedule_logger().exception(e)
        finally:
            return remaining_cores, remaining_memory

    @classmethod
    @DB.connection_context()
    def get_engine_registration_info(cls, engine_type, engine_name) -> EngineRegistry:
        engines = EngineRegistry.select().where(EngineRegistry.f_engine_type == engine_type,
                                                EngineRegistry.f_engine_name == engine_name)
        if engines:
            return engines[0]
        else:
            return None
