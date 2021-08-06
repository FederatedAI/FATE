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
import sys
import os
from fate_arch.common import engine_utils
from fate_arch.computing import ComputingEngine
from fate_arch.common import EngineType
from fate_arch.common.base_utils import json_dumps, current_timestamp, fate_uuid
from fate_flow.utils.authentication_utils import data_authentication_check
from fate_arch.common.log import schedule_logger
from fate_flow.controller.task_controller import TaskController
from fate_flow.entity.run_status import JobStatus, EndStatus
from fate_flow.entity.run_parameters import RunParameters
from fate_flow.entity.component_provider import ComponentProvider
from fate_flow.entity.types import InputSearchType
from fate_flow.manager.resource_manager import ResourceManager
from fate_flow.operation.job_saver import JobSaver
from fate_flow.operation.job_tracker import Tracker
from fate_flow.settings import USE_DATA_AUTHENTICATION
from fate_flow.protobuf.python import pipeline_pb2
from fate_flow.runtime_config import RuntimeConfig
from fate_flow.settings import USE_AUTHENTICATION
from fate_flow import job_default_settings
from fate_flow.utils import job_utils, schedule_utils, data_utils
from fate_flow.component_env_utils import dsl_utils
from fate_flow.utils.authentication_utils import authentication_check
from fate_flow.operation.task_initializer import TaskInitializer
import subprocess


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
        schedule_logger(job_id).info('job parameters:{}'.format(job_parameters))
        dest_user = dsl_parser.get_job_parameters().get(role, {}).get(party_id, {}).get("user", '')
        user = {}
        src_party_id = int(job_info.get('src_party_id')) if job_info.get('src_party_id') else 0
        src_user = dsl_parser.get_job_parameters().get(job_info.get('src_role'), {}).get(src_party_id, {}).get("user", '')
        for _role, party_id_item in dsl_parser.get_job_parameters().items():
            user[_role] = {}
            for _party_id, _parameters in party_id_item.items():
                user[_role][_party_id] = _parameters.get("user", "")
        schedule_logger(job_id).info('job user:{}'.format(user))
        if USE_DATA_AUTHENTICATION:
            job_args = dsl_parser.get_args_input()
            schedule_logger(job_id).info('job args:{}'.format(job_args))
            dataset_dict = cls.get_dataset(False, role, party_id, runtime_conf.get("role"), job_args)
            dataset_list = []
            if dataset_dict.get(role, {}).get(party_id):
                for k, v in dataset_dict[role][party_id].items():
                    dataset_list.append({"namespace": v.split('.')[0], "table_name": v.split('.')[1]})
            data_authentication_check(src_role=job_info.get('src_role'), src_party_id=job_info.get('src_party_id'),
                                      src_user=src_user, dest_user=dest_user, dataset_list=dataset_list)
        job_parameters = RunParameters(**job_parameters)

        # save new job into db
        if role == job_info["initiator_role"] and party_id == job_info["initiator_party_id"]:
            is_initiator = True
        else:
            is_initiator = False
        job_info["status"] = JobStatus.READY
        job_info["user_id"] = dest_user
        job_info["src_user"] = src_user
        job_info["user"] = user
        # this party configuration
        job_info["role"] = role
        job_info["party_id"] = party_id
        job_info["is_initiator"] = is_initiator
        job_info["progress"] = 0
        cls.fill_party_specific_parameters(role=role,
                                           party_id=party_id,
                                           job_parameters=job_parameters)
        # update job parameters on party
        job_info["runtime_conf_on_party"]["job_parameters"] = job_parameters.to_dict()
        job_utils.save_job_conf(job_id=job_id,
                                role=role,
                                party_id=party_id,
                                job_dsl=dsl,
                                job_runtime_conf=runtime_conf,
                                job_runtime_conf_on_party=job_info["runtime_conf_on_party"],
                                train_runtime_conf=train_runtime_conf,
                                pipeline_dsl=None)

        cls.initialize_tasks(job_id=job_id, role=role, party_id=party_id, run_on_this_party=True,
                             initiator_role=job_info["initiator_role"], initiator_party_id=job_info["initiator_party_id"], job_parameters=job_parameters, dsl_parser=dsl_parser)
        roles = job_info['roles']
        cls.initialize_job_tracker(job_id=job_id, role=role, party_id=party_id,
                                   job_parameters=job_parameters, roles=roles, is_initiator=is_initiator, dsl_parser=dsl_parser)
        JobSaver.create_job(job_info=job_info)

    @classmethod
    def get_job_engines(cls, job_parameters: RunParameters):
        kwargs = {}
        for k in {EngineType.COMPUTING, EngineType.FEDERATION, EngineType.STORAGE}:
            kwargs[k] = getattr(job_parameters, f"{k}_engine", None)
        engines = engine_utils.engines_compatibility(
            work_mode=job_parameters.work_mode,
            backend=job_parameters.backend,
            federated_mode=job_parameters.federated_mode,
            **kwargs
        )
        for k in {EngineType.COMPUTING, EngineType.FEDERATION, EngineType.STORAGE}:
            setattr(job_parameters, f"{k}_engine", engines[k])
        job_parameters.federated_mode = engines["federated_mode"]

    @classmethod
    def create_common_job_parameters(cls, job_id, initiator_role, common_job_parameters: RunParameters):
        JobController.get_job_engines(job_parameters=common_job_parameters)
        JobController.fill_default_job_parameters(job_id=job_id, job_parameters=common_job_parameters)
        JobController.adapt_job_parameters(role=initiator_role, job_parameters=common_job_parameters, create_initiator_baseline=True)

    @classmethod
    def fill_party_specific_parameters(cls, role, party_id, job_parameters: RunParameters):
        cls.adapt_job_parameters(role=role, job_parameters=job_parameters)
        engines_info = cls.get_job_engines_address(job_parameters=job_parameters)
        cls.check_parameters(job_parameters=job_parameters,
                             role=role, party_id=party_id, engines_info=engines_info)

    @classmethod
    def fill_default_job_parameters(cls, job_id, job_parameters: RunParameters):
        keys = {"task_parallelism", "auto_retries", "auto_retry_delay", "federated_status_collect_type"}
        for key in keys:
            if hasattr(job_parameters, key) and getattr(job_parameters, key) is None:
                if hasattr(job_default_settings, key.upper()):
                    setattr(job_parameters, key, getattr(job_default_settings, key.upper()))
                else:
                    schedule_logger(job_id=job_id).warning(f"can not found {key} job parameter default value from job_default_settings")

    @classmethod
    def adapt_job_parameters(cls, role, job_parameters: RunParameters, create_initiator_baseline=False):
        ResourceManager.adapt_engine_parameters(
            role=role, job_parameters=job_parameters, create_initiator_baseline=create_initiator_baseline)
        if create_initiator_baseline and not job_parameters.computing_partitions:
            job_parameters.computing_partitions = job_parameters.adaptation_parameters[
                "task_cores_per_node"] * job_parameters.adaptation_parameters["task_nodes"]
        if not job_parameters.component_provider or not job_parameters.component_version:
            #todo: component type may be not from job parameters
            job_parameters.component_provider, job_parameters.component_version = job_utils.get_default_component_use(
                component_provider=job_parameters.component_provider)

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
    def gen_updated_parameters(cls, job_id, initiator_role, initiator_party_id, input_job_parameters, input_component_parameters):
        job_configuration = job_utils.get_job_configuration(job_id=job_id,
                                                            role=initiator_role,
                                                            party_id=initiator_party_id)
        updated_job_parameters = job_configuration.runtime_conf["job_parameters"]
        updated_component_parameters = job_configuration.runtime_conf["component_parameters"]
        if input_job_parameters:
            if input_job_parameters.get("common"):
                common_job_parameters = RunParameters(**input_job_parameters["common"])
                cls.create_common_job_parameters(job_id=job_id, initiator_role=initiator_role, common_job_parameters=common_job_parameters)
                for attr in {"model_id", "model_version"}:
                    setattr(common_job_parameters, attr, updated_job_parameters["common"].get(attr))
                updated_job_parameters["common"] = common_job_parameters.to_dict()
            # not support role
        updated_components = set()
        if input_component_parameters:
            if input_component_parameters.get("common"):
                if "common" not in updated_component_parameters:
                    updated_component_parameters["common"] = {}
                for name, parameters in input_component_parameters["common"].items():
                    updated_component_parameters["common"][name] = parameters
                    updated_components.add(name)
            if input_component_parameters.get("role"):
                if "role" not in updated_component_parameters:
                    updated_component_parameters["role"] = {}
                for _role, role_parties in input_component_parameters["role"].items():
                    updated_component_parameters["role"][_role] = updated_component_parameters["role"].get(_role, {})
                    for party_index, components in role_parties.items():
                        updated_component_parameters["role"][_role][party_index] = updated_component_parameters["role"][_role].get(party_index, {})
                        for name, parameters in components.items():
                            updated_component_parameters["role"][_role][party_index][name] = parameters
                            updated_components.add(name)
        return updated_job_parameters, updated_component_parameters, list(updated_components)

    @classmethod
    def update_parameter(cls, job_id, role, party_id, updated_parameters: dict):
        job_configuration = job_utils.get_job_configuration(job_id=job_id,
                                                            role=role,
                                                            party_id=party_id)
        job_parameters = updated_parameters.get("job_parameters")
        component_parameters = updated_parameters.get("component_parameters")
        if job_parameters:
            job_configuration.runtime_conf["job_parameters"] = job_parameters
            job_parameters = RunParameters(**job_parameters["common"])
            cls.fill_party_specific_parameters(role=role,
                                               party_id=party_id,
                                               job_parameters=job_parameters)
            job_configuration.runtime_conf_on_party["job_parameters"] = job_parameters.to_dict()
        if component_parameters:
            job_configuration.runtime_conf["component_parameters"] = component_parameters
            job_configuration.runtime_conf_on_party["component_parameters"] = component_parameters

        job_info = {}
        job_info["job_id"] = job_id
        job_info["role"] = role
        job_info["party_id"] = party_id
        job_info["runtime_conf"] = job_configuration.runtime_conf
        job_info["runtime_conf_on_party"] = job_configuration.runtime_conf_on_party
        JobSaver.update_job(job_info)

    @classmethod
    def initialize_tasks(cls, job_id, role, party_id, run_on_this_party, initiator_role, initiator_party_id, job_parameters: RunParameters, dsl_parser, component_name=None, task_version=None, auto_retries=None):
        common_task_info = {}
        common_task_info["job_id"] = job_id
        common_task_info["initiator_role"] = initiator_role
        common_task_info["initiator_party_id"] = initiator_party_id
        common_task_info["role"] = role
        common_task_info["party_id"] = party_id
        common_task_info["run_on_this_party"] = run_on_this_party
        common_task_info["federated_mode"] = job_parameters.federated_mode
        common_task_info["federated_status_collect_type"] = job_parameters.federated_status_collect_type
        common_task_info["auto_retries"] = auto_retries if auto_retries is not None else job_parameters.auto_retries
        common_task_info["auto_retry_delay"] = job_parameters.auto_retry_delay
        if task_version:
            common_task_info["task_version"] = task_version
        provider_group = dsl_utils.get_job_provider_group(dsl_parser=dsl_parser,
                                                          role=role,
                                                          party_id=party_id,
                                                          component_name=component_name)
        for group_key, group_info in provider_group.items():
            initialized_config = {}
            initialized_config.update(group_info)
            initialized_config["common_task_info"] = common_task_info
            cls.start_initializer(job_id=job_id,
                                  role=role,
                                  party_id=party_id,
                                  initialized_config=initialized_config)

    @classmethod
    def start_initializer(cls, job_id, role, party_id, initialized_config):
        initializer_id = fate_uuid()
        initialized_components = initialized_config["components"]
        party_id = str(party_id)
        schedule_logger(job_id).info('try to start job {} task initializer {} subprocess to initialize {} on {} {}'.format(job_id, initializer_id, initialized_components, role, party_id))
        initialize_dir = os.path.join(job_utils.get_job_directory(job_id=job_id), role, party_id, f"initialize_{initializer_id}")
        os.makedirs(initialize_dir, exist_ok=True)
        initialized_config_path = os.path.join(initialize_dir, f'initialized_config.json')
        with open(initialized_config_path, 'w') as fw:
            fw.write(json_dumps(initialized_config))

        process_cmd = [
            sys.executable,
            sys.modules[TaskInitializer.__module__].__file__,
            '-j', job_id,
            '-r', role,
            '-p', party_id,
            '-c', initialized_config_path,
            '--run_ip', RuntimeConfig.JOB_SERVER_HOST,
            '--job_server', f'{RuntimeConfig.JOB_SERVER_HOST}:{RuntimeConfig.HTTP_PORT}',
        ]
        log_dir = os.path.join(job_utils.get_job_log_directory(job_id=job_id), role, party_id, "initialize", initializer_id)
        provider = ComponentProvider(**initialized_config["provider"])
        p = job_utils.run_subprocess(job_id=job_id, config_dir=initialize_dir, process_cmd=process_cmd, extra_env=provider.env, log_dir=log_dir, cwd_dir=initialize_dir)
        schedule_logger(job_id).info('job {} task initializer {} on {} {} subprocess pid {} is ready'.format(job_id, initializer_id, role, party_id, p.pid))
        try:
            p.communicate(timeout=5)
            # return code always 0 because of server wait_child_process, can not use to check
            st = JobSaver.check_task(job_id=job_id, role=role, party_id=party_id, components=initialized_components)
            schedule_logger(job_id).info('job {} initialize {} on {} {} {}'.format(job_id, initialized_components, role, party_id, "successfully" if st else "failed"))
            #todo: check
            """
            if not st:
                raise Exception(job_utils.get_subprocess_std(log_dir=log_dir))
            """
        except subprocess.TimeoutExpired as e:
            err = f"job {job_id} task initializer {initializer_id} on {role} {party_id} subprocess pid {p.pid} run timeout"
            schedule_logger(job_id).exception(err, e)
            p.kill()
            raise Exception(err)

    @classmethod
    def initialize_job_tracker(cls, job_id, role, party_id, job_parameters: RunParameters, roles, is_initiator, dsl_parser):
        tracker = Tracker(job_id=job_id, role=role, party_id=party_id,
                          model_id=job_parameters.model_id,
                          model_version=job_parameters.model_version,
                          job_parameters=job_parameters)
        if job_parameters.job_type != "predict":
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
        for _role, _role_party_args in job_args.items():
            if _role == "dsl_version":
                continue
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
                                    search_type = data_utils.get_input_search_type(parameters=_data_location)
                                    if search_type == InputSearchType.TABLE_INFO:
                                        dataset[_role][_party_id][key] = '{}.{}'.format(_data_location['namespace'], _data_location['name'])
                                    elif search_type == InputSearchType.JOB_COMPONENT_OUTPUT:
                                        dataset[_role][_party_id][key] = '{}.{}.{}'.format(_data_location['job_id'], _data_location['component_name'], _data_location['data'])
                                    else:
                                        dataset[_role][_party_id][key] = "unknown"
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
        job_configuration = job_utils.get_job_configuration(job_id=job_id, role=role,
                                                            party_id=party_id)
        runtime_conf_on_party = job_configuration.runtime_conf_on_party
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
        dsl_parser = schedule_utils.get_job_dsl_parser(dsl=job_configuration.dsl,
                                                       runtime_conf=job_configuration.runtime_conf,
                                                       train_runtime_conf=job_configuration.train_runtime_conf)
        predict_dsl = dsl_parser.get_predict_dsl(role=role)
        pipeline = pipeline_pb2.Pipeline()
        pipeline.inference_dsl = json_dumps(predict_dsl, byte=True)
        pipeline.train_dsl = json_dumps(job_configuration.dsl, byte=True)
        pipeline.train_runtime_conf = json_dumps(job_configuration.train_runtime_conf, byte=True)
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
                          model_id=model_id, model_version=model_version, job_parameters=RunParameters(**job_parameters))
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
