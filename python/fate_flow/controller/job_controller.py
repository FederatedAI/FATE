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
from fate_flow.utils.authentication_utils import authentication_check
from federatedml.protobuf.generated import pipeline_pb2
from fate_arch.common.log import schedule_logger
from fate_arch.common import EngineType, string_utils
from fate_flow.entity.types import JobStatus, EndStatus, RunParameters
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.operation.job_tracker import Tracker
from fate_flow.settings import USE_AUTHENTICATION, DEFAULT_TASK_PARALLELISM, DEFAULT_FEDERATED_STATUS_COLLECT_TYPE
from fate_flow.utils import job_utils, schedule_utils, data_utils
from fate_flow.operation.job_saver import JobSaver
from fate_arch.common.base_utils import json_dumps, current_timestamp
from fate_flow.controller.task_controller import TaskController
from fate_flow.manager.resource_manager import ResourceManager
from fate_arch.common import WorkMode, Backend, StandaloneBackend
from fate_arch.common import FederatedMode
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.storage import StorageEngine
from fate_flow.settings import WORK_MODE


class JobController(object):
    @classmethod
    def create_job(cls, job_id, role, party_id, job_info):
        # parse job configuration
        dsl = job_info['dsl']
        runtime_conf = job_info['runtime_conf']
        train_runtime_conf = job_info['train_runtime_conf']
        if USE_AUTHENTICATION:
            authentication_check(src_role=job_info.get('src_role', None), src_party_id=job_info.get('src_party_id', None),
                                 dsl=dsl, runtime_conf=runtime_conf, role=role, party_id=party_id)

        dsl_parser = schedule_utils.get_job_dsl_parser(dsl=dsl,
                                                       runtime_conf=runtime_conf,
                                                       train_runtime_conf=train_runtime_conf)
        job_parameters = dsl_parser.get_job_parameters().get(role, {}).get(party_id, {})
        schedule_logger(job_id).info(
            'job parameters:{}'.format(job_parameters))
        job_parameters = RunParameters(**job_parameters)

        # adjust parameters if work_mode is different from remote party
        if job_parameters.work_mode != WORK_MODE:
            schedule_logger(job_id).info(
                'received job config with different work mode defined in settings, update the backend compatibility forcibly')
            job_parameters.work_mode = WORK_MODE
            JobController.backend_compatibility(
                job_parameters=job_parameters, force_update=True)

            schedule_logger(job_id).info(
                'job parameters after forced updated, job parameters:{}'.format(job_parameters.to_dict()))

        # save new job into db
        if role == job_info["initiator_role"] and party_id == job_info["initiator_party_id"]:
            is_initiator = True
        else:
            is_initiator = False
        job_info["status"] = JobStatus.WAITING
        # this party configuration
        job_info["role"] = role
        job_info["party_id"] = party_id
        job_info["is_initiator"] = is_initiator
        job_info["progress"] = 0
        cls.adapt_job_parameters(role=role, job_parameters=job_parameters)
        engines_info = cls.get_job_engines_address(
            job_parameters=job_parameters)
        cls.check_parameters(job_parameters=job_parameters,
                             role=role, party_id=party_id, engines_info=engines_info)
        job_info["runtime_conf_on_party"]["job_parameters"] = job_parameters.to_dict()
        job_utils.save_job_conf(job_id=job_id,
                                role=role,
                                job_dsl=dsl,
                                job_runtime_conf=runtime_conf,
                                job_runtime_conf_on_party=job_info["runtime_conf_on_party"],
                                train_runtime_conf=train_runtime_conf,
                                pipeline_dsl=None)

        cls.initialize_tasks(job_id=job_id, role=role, party_id=party_id, run_on_this_party=True,
                             initiator_role=job_info["initiator_role"], initiator_party_id=job_info["initiator_party_id"], job_parameters=job_parameters, dsl_parser=dsl_parser)
        job_parameters = job_info['runtime_conf_on_party']['job_parameters']
        roles = job_info['roles']
        cls.initialize_job_tracker(job_id=job_id, role=role, party_id=party_id,
                                   job_parameters=job_parameters, roles=roles, is_initiator=is_initiator, dsl_parser=dsl_parser)
        JobSaver.create_job(job_info=job_info)

    @classmethod
    def backend_compatibility(cls, job_parameters: RunParameters, force_update: bool = False):
        # compatible with previous 1.5 versions
        if job_parameters.computing_engine is None or job_parameters.federation_engine is None or force_update:
            if job_parameters.work_mode is None or job_parameters.backend is None:
                raise RuntimeError("unable to find compatible backend engines")
            work_mode = WorkMode(job_parameters.work_mode)

            if work_mode == WorkMode.STANDALONE:
                backend = StandaloneBackend(job_parameters.backend)
                if backend == StandaloneBackend.STANDALONE_PURE:
                    job_parameters.computing_engine = ComputingEngine.STANDALONE
                    job_parameters.federation_engine = FederationEngine.STANDALONE
                    job_parameters.storage_engine = StorageEngine.STANDALONE
                elif backend == StandaloneBackend.STANDALONE_RABBITMQ:
                    job_parameters.computing_engine = ComputingEngine.SPARK
                    job_parameters.federation_engine = FederationEngine.RABBITMQ
                    job_parameters.storage_engine = StorageEngine.LOCAL

                    # add mq info
                    if job_parameters.federation_info == None:
                        federation_info = {}
                        federation_info['union_name'] = string_utils.random_string(
                            4)
                        federation_info['policy_id'] = string_utils.random_string(
                            10)
                        job_parameters.federation_info = federation_info
                elif backend == StandaloneBackend.STANDALONE_PULSAR:
                    job_parameters.computing_engine = ComputingEngine.SPARK
                    job_parameters.federation_engine = FederationEngine.PULSAR
                    job_parameters.storage_engine = StorageEngine.LOCAL

            if work_mode == WorkMode.CLUSTER:
                backend = Backend(job_parameters.backend)
                if backend == Backend.EGGROLL:
                    job_parameters.computing_engine = ComputingEngine.EGGROLL
                    job_parameters.federation_engine = FederationEngine.EGGROLL
                    job_parameters.storage_engine = StorageEngine.EGGROLL
                elif backend == Backend.SPARK_PULSAR:
                    job_parameters.computing_engine = ComputingEngine.SPARK
                    job_parameters.federation_engine = FederationEngine.PULSAR
                    job_parameters.storage_engine = StorageEngine.HDFS
                elif backend == Backend.SPARK_RABBITMQ:
                    job_parameters.computing_engine = ComputingEngine.SPARK
                    job_parameters.federation_engine = FederationEngine.RABBITMQ
                    job_parameters.storage_engine = StorageEngine.HDFS
                    # add mq info
                    if job_parameters.federation_info == None:
                        federation_info = {}
                        federation_info['union_name'] = string_utils.random_string(
                            4)
                        federation_info['policy_id'] = string_utils.random_string(
                            10)
                        job_parameters.federation_info = federation_inf

        if job_parameters.federated_mode is None:
            if job_parameters.computing_engine in [ComputingEngine.EGGROLL, ComputingEngine.SPARK]:
                job_parameters.federated_mode = FederatedMode.MULTIPLE
            elif job_parameters.computing_engine in [ComputingEngine.STANDALONE]:
                job_parameters.federated_mode = FederatedMode.SINGLE

    @classmethod
    def adapt_job_parameters(cls, role, job_parameters: RunParameters, create_initiator_baseline=False):
        ResourceManager.adapt_engine_parameters(
            role=role, job_parameters=job_parameters, create_initiator_baseline=create_initiator_baseline)
        if create_initiator_baseline:
            if job_parameters.task_parallelism is None:
                job_parameters.task_parallelism = DEFAULT_TASK_PARALLELISM
            if job_parameters.federated_status_collect_type is None:
                job_parameters.federated_status_collect_type = DEFAULT_FEDERATED_STATUS_COLLECT_TYPE
        if create_initiator_baseline and not job_parameters.computing_partitions:
            job_parameters.computing_partitions = job_parameters.adaptation_parameters[
                "task_cores_per_node"] * job_parameters.adaptation_parameters["task_nodes"]

    @classmethod
    def get_job_engines_address(cls, job_parameters: RunParameters):
        engines_info = {}
        engine_list = [
            (EngineType.COMPUTING, job_parameters.computing_engine),
            (EngineType.FEDERATION, job_parameters.federation_engine),
            (EngineType.STORAGE, job_parameters.storage_engine)
        ]
        for engine_type, engine_name in engine_list:
            engine_info = ResourceManager.get_engine_registration_info(
                engine_type=engine_type, engine_name=engine_name)
            job_parameters.engines_address[engine_type] = engine_info.f_engine_config
            engines_info[engine_type] = engine_info
        return engines_info

    @classmethod
    def check_parameters(cls, job_parameters: RunParameters, role, party_id, engines_info):
        status, cores_submit, max_cores_per_job = ResourceManager.check_resource_apply(
            job_parameters=job_parameters, role=role, party_id=party_id, engines_info=engines_info)
        if not status:
            msg = ""
            msg2 = "default value is fate_flow/settings.py#DEFAULT_TASK_CORES_PER_NODE, refer fate_flow/examples/simple_hetero_lr_job_conf.json"
            if job_parameters.computing_engine in {ComputingEngine.EGGROLL, ComputingEngine.STANDALONE}:
                msg = "please use task_cores job parameters to set request task cores or you can customize it with eggroll_run job parameters"
            elif job_parameters.computing_engine in {ComputingEngine.SPARK}:
                msg = "please use task_cores job parameters to set request task cores or you can customize it with spark_run job parameters"
            raise RuntimeError(
                f"max cores per job is {max_cores_per_job} base on (fate_flow/settings#MAX_CORES_PERCENT_PER_JOB * conf/service_conf.yaml#nodes * conf/service_conf.yaml#cores_per_node), expect {cores_submit} cores, {msg}, {msg2}")

    @classmethod
    def initialize_tasks(cls, job_id, role, party_id, run_on_this_party, initiator_role, initiator_party_id, job_parameters: RunParameters, dsl_parser, component_name=None, task_version=None):
        common_task_info = {}
        common_task_info["job_id"] = job_id
        common_task_info["initiator_role"] = initiator_role
        common_task_info["initiator_party_id"] = initiator_party_id
        common_task_info["role"] = role
        common_task_info["party_id"] = party_id
        common_task_info["federated_mode"] = job_parameters.federated_mode
        common_task_info["federated_status_collect_type"] = job_parameters.federated_status_collect_type

        if task_version:
            common_task_info["task_version"] = task_version
        if not component_name:
            components = dsl_parser.get_topology_components()
        else:
            components = [dsl_parser.get_component_info(
                component_name=component_name)]
        for component in components:
            component_parameters = component.get_role_parameters()
            for parameters_on_party in component_parameters.get(common_task_info["role"], []):
                if parameters_on_party.get('local', {}).get('party_id') == common_task_info["party_id"]:
                    task_info = {}
                    task_info.update(common_task_info)
                    task_info["component_name"] = component.get_name()
                    TaskController.create_task(
                        role=role, party_id=party_id, run_on_this_party=run_on_this_party, task_info=task_info)

    @classmethod
    def initialize_job_tracker(cls, job_id, role, party_id, job_parameters, roles, is_initiator, dsl_parser):
        tracker = Tracker(job_id=job_id, role=role, party_id=party_id,
                          model_id=job_parameters["model_id"],
                          model_version=job_parameters["model_version"])
        if job_parameters.get("job_type", "") != "predict":
            tracker.init_pipelined_model()
        partner = {}
        show_role = {}
        for _role, _role_party in roles.items():
            if is_initiator or _role == role:
                show_role[_role] = show_role.get(_role, [])
                for _party_id in _role_party:
                    if is_initiator or _party_id == party_id:
                        show_role[_role].append(_party_id)

            if _role != role:
                partner[_role] = partner.get(_role, [])
                partner[_role].extend(_role_party)
            else:
                for _party_id in _role_party:
                    if _party_id != party_id:
                        partner[_role] = partner.get(_role, [])
                        partner[_role].append(_party_id)

        job_args = dsl_parser.get_args_input()
        dataset = cls.get_dataset(
            is_initiator, role, party_id, roles, job_args)
        tracker.log_job_view(
            {'partner': partner, 'dataset': dataset, 'roles': show_role})

    @classmethod
    def get_dataset(cls, is_initiator, role, party_id, roles, job_args):
        dataset = {}
        dsl_version = 1
        if job_args.get('dsl_version'):
            if job_args.get('dsl_version') == 2:
                dsl_version = 2
            job_args.pop('dsl_version')
        for _role, _role_party_args in job_args.items():
            if is_initiator or _role == role:
                for _party_index in range(len(_role_party_args)):
                    _party_id = roles[_role][_party_index]
                    if is_initiator or _party_id == party_id:
                        dataset[_role] = dataset.get(_role, {})
                        dataset[_role][_party_id] = dataset[_role].get(
                            _party_id, {})
                        if dsl_version == 1:
                            for _data_type, _data_location in _role_party_args[_party_index]['args']['data'].items():
                                dataset[_role][_party_id][_data_type] = '{}.{}'.format(
                                    _data_location['namespace'], _data_location['name'])
                        else:
                            for key in _role_party_args[_party_index].keys():
                                for _data_type, _data_location in _role_party_args[_party_index][key].items():
                                    dataset[_role][_party_id][key] = '{}.{}'.format(
                                        _data_location['namespace'], _data_location['name'])
        return dataset

    @classmethod
    def query_job_input_args(cls, input_data, role, party_id):
        min_partition = data_utils.get_input_data_min_partitions(
            input_data, role, party_id)
        return {'min_input_data_partition': min_partition}

    @classmethod
    def start_job(cls, job_id, role, party_id, extra_info=None):
        schedule_logger(job_id=job_id).info(
            f"try to start job {job_id} on {role} {party_id}")
        job_info = {
            "job_id": job_id,
            "role": role,
            "party_id": party_id,
            "status": JobStatus.RUNNING,
            "start_time": current_timestamp()
        }
        if extra_info:
            schedule_logger(job_id=job_id).info(f"extra info: {extra_info}")
            job_info.update(extra_info)
        cls.update_job_status(job_info=job_info)
        cls.update_job(job_info=job_info)
        schedule_logger(job_id=job_id).info(
            f"start job {job_id} on {role} {party_id} successfully")

    @classmethod
    def update_job(cls, job_info):
        """
        Save to local database
        :param job_info:
        :return:
        """
        return JobSaver.update_job(job_info=job_info)

    @classmethod
    def update_job_status(cls, job_info):
        update_status = JobSaver.update_job_status(job_info=job_info)
        if update_status and EndStatus.contains(job_info.get("status")):
            ResourceManager.return_job_resource(
                job_id=job_info["job_id"], role=job_info["role"], party_id=job_info["party_id"])
        return update_status

    @classmethod
    def stop_jobs(cls, job_id, stop_status, role=None, party_id=None):
        if role and party_id:
            jobs = JobSaver.query_job(
                job_id=job_id, role=role, party_id=party_id)
        else:
            jobs = JobSaver.query_job(job_id=job_id)
        kill_status = True
        kill_details = {}
        for job in jobs:
            kill_job_status, kill_job_details = cls.stop_job(
                job=job, stop_status=stop_status)
            kill_status = kill_status & kill_job_status
            kill_details[job_id] = kill_job_details
        return kill_status, kill_details

    @classmethod
    def stop_job(cls, job, stop_status):
        tasks = JobSaver.query_task(
            job_id=job.f_job_id, role=job.f_role, party_id=job.f_party_id, reverse=True)
        kill_status = True
        kill_details = {}
        for task in tasks:
            kill_task_status = TaskController.stop_task(
                task=task, stop_status=stop_status)
            kill_status = kill_status & kill_task_status
            kill_details[task.f_task_id] = 'success' if kill_task_status else 'failed'
        if kill_status:
            job_info = job.to_human_model_dict(only_primary_with=["status"])
            job_info["status"] = stop_status
            JobController.update_job_status(job_info)
        return kill_status, kill_details
        # Job status depends on the final operation result and initiator calculate

    @classmethod
    def save_pipelined_model(cls, job_id, role, party_id):
        schedule_logger(job_id).info(
            'job {} on {} {} start to save pipeline'.format(job_id, role, party_id))
        job_dsl, job_runtime_conf, runtime_conf_on_party, train_runtime_conf = job_utils.get_job_configuration(job_id=job_id, role=role,
                                                                                                               party_id=party_id)
        job_parameters = runtime_conf_on_party.get('job_parameters', {})
        if role in job_parameters.get("assistant_role", []):
            return
        model_id = job_parameters['model_id']
        model_version = job_parameters['model_version']
        job_type = job_parameters.get('job_type', '')
        work_mode = job_parameters['work_mode']
        roles = runtime_conf_on_party['role']
        initiator_role = runtime_conf_on_party['initiator']['role']
        initiator_party_id = runtime_conf_on_party['initiator']['party_id']
        if job_type == 'predict':
            return
        dag = schedule_utils.get_job_dsl_parser(dsl=job_dsl,
                                                runtime_conf=job_runtime_conf,
                                                train_runtime_conf=train_runtime_conf)
        predict_dsl = dag.get_predict_dsl(role=role)
        pipeline = pipeline_pb2.Pipeline()
        pipeline.inference_dsl = json_dumps(predict_dsl, byte=True)
        pipeline.train_dsl = json_dumps(job_dsl, byte=True)
        pipeline.train_runtime_conf = json_dumps(job_runtime_conf, byte=True)
        pipeline.fate_version = RuntimeConfig.get_env("FATE")
        pipeline.model_id = model_id
        pipeline.model_version = model_version

        pipeline.parent = True
        pipeline.loaded_times = 0
        pipeline.roles = json_dumps(roles, byte=True)
        pipeline.work_mode = work_mode
        pipeline.initiator_role = initiator_role
        pipeline.initiator_party_id = initiator_party_id
        pipeline.runtime_conf_on_party = json_dumps(
            runtime_conf_on_party, byte=True)
        pipeline.parent_info = json_dumps({}, byte=True)

        tracker = Tracker(job_id=job_id, role=role, party_id=party_id,
                          model_id=model_id, model_version=model_version)
        tracker.save_pipelined_model(pipelined_buffer_object=pipeline)
        if role != 'local':
            tracker.save_machine_learning_model_info()
        schedule_logger(job_id).info(
            'job {} on {} {} save pipeline successfully'.format(job_id, role, party_id))

    @classmethod
    def clean_job(cls, job_id, role, party_id, roles):
        schedule_logger(job_id).info(
            'Job {} on {} {} start to clean'.format(job_id, role, party_id))
        # todo
        schedule_logger(job_id).info(
            'job {} on {} {} clean done'.format(job_id, role, party_id))
