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
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.storage import StorageEngine
from fate_arch.common import base_utils
from fate_arch.common.conf_utils import get_base_config
from fate_arch.common.log import schedule_logger
from fate_arch.common import EngineType
from fate_flow.db.db_models import DB, BackendRegistry, Job
from fate_flow.entity.types import ResourceOperation, RunParameters
from fate_flow.settings import stat_logger, STANDALONE_BACKEND_VIRTUAL_CORES_PER_NODE
from fate_flow.utils import job_utils


class ResourceManager(object):
    @classmethod
    def initialize(cls):
        # initialize default backend
        # storage backend is for component output data
        default_backends = {
            EngineType.COMPUTING: [ComputingEngine.EGGROLL, ComputingEngine.SPARK],
            EngineType.FEDERATION: [FederationEngine.EGGROLL, FederationEngine.MQ],
            EngineType.STORAGE: [StorageEngine.EGGROLL, StorageEngine.HDFS]
        }
        for engine_type, engines_name in default_backends.items():
            for engine_name in engines_name:
                engine_info = get_base_config(engine_name, {})
                if engine_info:
                    engine_info["engine"] = engine_name
                    cls._initialize_backend(engine_type=engine_type, engine_id=f"DEFAULT_{engine_name}", engine_info=engine_info)
        # initialize standalone backend
        for engine_type in default_backends.keys():
            engine_name = "STANDALONE"
            engine_info = {
                "engine": engine_name,
                "nodes": 1,
                "cores_per_node": STANDALONE_BACKEND_VIRTUAL_CORES_PER_NODE,
            }
            cls._initialize_backend(engine_type=engine_type, engine_id=f"DEFAULT_{engine_name}", engine_info=engine_info)
        # initialize multi backend if has
        for engine_type in [EngineType.COMPUTING, EngineType.FEDERATION, EngineType.STORAGE]:
            for engine_id, engine_info in get_base_config(engine_type, {}).items():
                cls._initialize_backend(engine_type=engine_type, engine_id=engine_id, engine_info=engine_info)

    @classmethod
    @DB.connection_context()
    def _initialize_backend(cls, engine_type, engine_id, engine_info):
        nodes = engine_info.get("nodes", 1)
        cores = engine_info.get("cores_per_node", 0) * nodes
        memory = engine_info.get("memory_per_node", 0) * nodes
        engine_name = engine_info.get("engine", "UNKNOWN")
        engine_address = engine_info.get("address", {})
        filters = [BackendRegistry.f_engine_id == engine_id, BackendRegistry.f_engine_type == engine_type]
        resources = BackendRegistry.select().where(*filters)
        if resources:
            resource = resources[0]
            update_fields = {}
            update_fields[BackendRegistry.f_engine_address] = engine_address
            update_fields[BackendRegistry.f_cores] = cores
            update_fields[BackendRegistry.f_memory] = memory
            update_fields[BackendRegistry.f_remaining_cores] = BackendRegistry.f_remaining_cores + (cores - resource.f_cores)
            update_fields[BackendRegistry.f_remaining_memory] = BackendRegistry.f_remaining_memory + (memory - resource.f_memory)
            update_fields[BackendRegistry.f_nodes] = nodes
            operate = BackendRegistry.update(update_fields).where(*filters)
            update_status = operate.execute() > 0
            if update_status:
                stat_logger.info(f"update {engine_type} engine {engine_id} registration information")
            else:
                stat_logger.info(f"update {engine_type} engine {engine_id} registration information takes no effect")
        else:
            resource = BackendRegistry()
            resource.f_create_time = base_utils.current_timestamp()
            resource.f_engine_id = engine_id
            resource.f_engine_name = engine_name
            resource.f_engine_type = engine_type
            resource.f_engine_address = engine_address
            resource.f_cores = cores
            resource.f_memory = memory
            resource.f_remaining_cores = cores
            resource.f_remaining_memory = memory
            resource.f_nodes = nodes
            try:
                resource.save(force_insert=True)
            except Exception as e:
                stat_logger.warning(e)
            stat_logger.info(f"create {engine_type} engine {engine_id} registration information")

    @classmethod
    @DB.connection_context()
    def apply_for_job_resource(cls, job_id, role, party_id):
        engine_id, cores, memory = cls.calculate_job_resource(job_id=job_id, role=role, party_id=party_id)
        apply_status, remaining_cores, remaining_memory = cls.update_resource(model=BackendRegistry,
                                                                              cores=cores,
                                                                              memory=memory,
                                                                              operation_type=ResourceOperation.APPLY,
                                                                              engine_type=EngineType.COMPUTING,
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
            operate = Job.update(update_fields).where(*filter_fields)
            update_status = operate.execute() > 0
            if update_status:
                schedule_logger(job_id=job_id).info(f"apply job {job_id} resource(cores {cores} memory {memory}) on {role} {party_id} successfully")
                return True
            else:
                schedule_logger(job_id=job_id).info(
                    f"save apply job {job_id} resource on {role} {party_id} record failed, rollback...")
                cls.return_job_resource(job_id=job_id, role=role, party_id=party_id)
                return False
        else:
            schedule_logger(job_id=job_id).info(f"apply job {job_id} resource(cores {cores} memory {memory}) on {role} {party_id} failed, remaining_cores: {remaining_cores}, remaining_memory: {remaining_memory}")
            return False

    @classmethod
    @DB.connection_context()
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
        operate = Job.update(update_fields).where(*filter_fields)
        update_status = operate.execute() > 0
        if update_status:
            return_status, remaining_cores, remaining_memory = cls.update_resource(model=BackendRegistry,
                                                                                   cores=cores,
                                                                                   memory=memory,
                                                                                   operation_type=ResourceOperation.RETURN,
                                                                                   engine_type=EngineType.COMPUTING,
                                                                                   engine_id=engine_id,
                                                                                   )
            if return_status:
                schedule_logger(job_id=job_id).info(f"return job {job_id} resource(cores {cores} memory {memory}) on {role} {party_id} successfully")
                return True
            else:
                schedule_logger(job_id=job_id).info(f"return job {job_id} resource(cores {cores} memory {memory}) on {role} {party_id} failed, remaining_cores: {remaining_cores}, remaining_memory: {remaining_memory}")
                return False
        else:
            schedule_logger(job_id=job_id).info(f"save return job {job_id} resource on {role} {party_id} record failed, do not return resource")
            return False

    @classmethod
    def calculate_job_resource(cls, job_id, role, party_id):
        dsl, runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=job_id,
                                                                                role=role,
                                                                                party_id=party_id)
        run_parameters = RunParameters(**runtime_conf["job_parameters"])
        cores = run_parameters.task_cores_per_node * run_parameters.task_nodes * run_parameters.task_parallelism
        memory = run_parameters.task_memory_per_node * run_parameters.task_nodes * run_parameters.task_parallelism
        computing_backend_info = cls.get_backend_registration_info(engine_type=EngineType.COMPUTING, engine_id=run_parameters.computing_backend)
        if computing_backend_info.f_engine_name in {ComputingEngine.EGGROLL, ComputingEngine.STANDALONE}:
            memory = 0
        return run_parameters.computing_backend, cores, memory

    @classmethod
    def calculate_task_resource(cls, task_info):
        dsl, runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=task_info["job_id"],
                                                                                role=task_info["role"],
                                                                                party_id=task_info["party_id"])
        run_parameters = RunParameters(**runtime_conf["job_parameters"])
        cores_per_task = run_parameters.task_cores_per_node * run_parameters.task_nodes
        memory_per_task = run_parameters.task_memory_per_node * run_parameters.task_nodes
        computing_backend_info = cls.get_backend_registration_info(engine_type=EngineType.COMPUTING, engine_id=run_parameters.computing_backend)
        if computing_backend_info.f_engine_name in {ComputingEngine.EGGROLL, ComputingEngine.STANDALONE}:
            memory_per_task = 0
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

        schedule_logger(job_id=task_info["job_id"]).info(
            "task {} {} try {} resource successfully".format(task_info["task_id"],
                                                             task_info["task_version"], operation_type))

        update_status, remaining_cores, remaining_memory = cls.update_resource(model=Job,
                                                                               cores=cores_per_task,
                                                                               memory=memory_per_task,
                                                                               operation_type=operation_type,
                                                                               job_id=task_info["job_id"],
                                                                               role=task_info["role"],
                                                                               party_id=task_info["party_id"],
                                                                               )
        if update_status:
            schedule_logger(job_id=task_info["job_id"]).info(
                "task {} {} {} resource successfully".format(task_info["task_id"],
                                                             task_info["task_version"], operation_type))
        else:
            schedule_logger(job_id=task_info["job_id"]).info(
                "task {} {} {} resource failed".format(task_info["task_id"],
                                                       task_info["task_version"], operation_type))
        return update_status

    @classmethod
    @DB.connection_context()
    def update_resource(cls, model, cores, memory, operation_type, **kwargs):
        filters = []
        primary_filters = []
        for p_k in model.get_primary_keys_name():
            primary_filters.append(operator.attrgetter(p_k)(model) == kwargs[p_k.lstrip("f").lstrip("_")])
        filters.extend(primary_filters)
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
        update_status = operate.execute() > 0
        if not update_status:
            objs = model.select(model.f_remaining_cores, model.f_remaining_memory).where(*primary_filters)
            remaining_cores, remaining_memory = objs[0].f_remaining_cores, objs[0].f_remaining_memory
        else:
            remaining_cores, remaining_memory = None, None
        return update_status, remaining_cores, remaining_memory

    @classmethod
    @DB.connection_context()
    def get_backend_registration_info(cls, engine_type, engine_id):
        engines = BackendRegistry.select().where(BackendRegistry.f_engine_type == engine_type, BackendRegistry.f_engine_id == engine_id)
        if engines:
            return engines[0]
        else:
            return None
