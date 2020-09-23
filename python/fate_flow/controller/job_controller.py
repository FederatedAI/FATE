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
from fate_arch.common import EngineType
from fate_flow.entity.types import JobStatus, EndStatus, RunParameters
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.operation import Tracker
from fate_flow.settings import USE_AUTHENTICATION, MAX_CORES_PERCENT_PER_JOB
from fate_flow.utils import job_utils, schedule_utils, data_utils
from fate_flow.operation import JobSaver, JobQueue
from fate_arch.common.base_utils import json_dumps, current_timestamp
from fate_flow.controller import TaskController
from fate_flow.manager import ResourceManager


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
        job_parameters = RunParameters(**runtime_conf['job_parameters'])
        job_initiator = runtime_conf['initiator']

        dsl_parser = schedule_utils.get_job_dsl_parser(dsl=dsl,
                                                       runtime_conf=runtime_conf,
                                                       train_runtime_conf=train_runtime_conf)

        # save new job into db
        if role == job_initiator['role'] and party_id == job_initiator['party_id']:
            is_initiator = True
        else:
            is_initiator = False
        job_info["status"] = JobStatus.WAITING
        roles = job_info['roles']
        # this party configuration
        job_info["role"] = role
        job_info["party_id"] = party_id
        job_info["is_initiator"] = is_initiator
        job_info["progress"] = 0
        cls.get_job_engines_address(job_parameters=job_parameters)
        cls.special_role_parameters(role=role, job_parameters=job_parameters)
        runtime_conf["job_parameters"] = job_parameters.to_dict()

        JobSaver.create_job(job_info=job_info)
        job_utils.save_job_conf(job_id=job_id,
                                job_dsl=dsl,
                                job_runtime_conf=runtime_conf,
                                train_runtime_conf=train_runtime_conf,
                                pipeline_dsl=None)

        cls.initialize_tasks(job_id, role, party_id, True, job_initiator, job_parameters, dsl_parser)
        cls.initialize_job_tracker(job_id=job_id, role=role, party_id=party_id, job_info=job_info, is_initiator=is_initiator, dsl_parser=dsl_parser)

    @classmethod
    def get_job_engines_address(cls, job_parameters: RunParameters):
        backend_info = ResourceManager.get_backend_registration_info(engine_type=EngineType.COMPUTING, engine_name=job_parameters.computing_engine)
        job_parameters.engines_address[EngineType.COMPUTING] = backend_info.f_engine_address
        backend_info = ResourceManager.get_backend_registration_info(engine_type=EngineType.FEDERATION, engine_name=job_parameters.federation_engine)
        job_parameters.engines_address[EngineType.FEDERATION] = backend_info.f_engine_address
        backend_info = ResourceManager.get_backend_registration_info(engine_type=EngineType.STORAGE, engine_name=job_parameters.storage_engine)
        job_parameters.engines_address[EngineType.STORAGE] = backend_info.f_engine_address

    @classmethod
    def special_role_parameters(cls, role, job_parameters: RunParameters):
        if role == "arbiter":
            job_parameters.task_nodes = 1
            job_parameters.task_parallelism = 1
            job_parameters.task_cores_per_node = 1

    @classmethod
    def initialize_tasks(cls, job_id, role, party_id, run_on, job_initiator, job_parameters: RunParameters, dsl_parser, component_name=None, task_version=None):
        common_task_info = {}
        common_task_info["job_id"] = job_id
        common_task_info["initiator_role"] = job_initiator['role']
        common_task_info["initiator_party_id"] = job_initiator['party_id']
        common_task_info["role"] = role
        common_task_info["party_id"] = party_id
        common_task_info["federated_mode"] = job_parameters.federated_mode
        common_task_info["federated_status_collect_type"] = job_parameters.federated_status_collect_type
        if task_version:
            common_task_info["task_version"] = task_version
        if not component_name:
            components = dsl_parser.get_topology_components()
        else:
            components = [dsl_parser.get_component_info(component_name=component_name)]
        for component in components:
            component_parameters = component.get_role_parameters()
            for parameters_on_party in component_parameters.get(common_task_info["role"], []):
                if parameters_on_party.get('local', {}).get('party_id') == common_task_info["party_id"]:
                    task_info = {}
                    task_info.update(common_task_info)
                    task_info["component_name"] = component.get_name()
                    TaskController.create_task(role=role, party_id=party_id, run_on=run_on, task_info=task_info)

    @classmethod
    def initialize_job_tracker(cls, job_id, role, party_id, job_info, is_initiator, dsl_parser):
        job_parameters = job_info['runtime_conf']['job_parameters']
        roles = job_info['roles']
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
        dataset = cls.get_dataset(is_initiator, role, party_id, roles, job_args)
        tracker.log_job_view({'partner': partner, 'dataset': dataset, 'roles': show_role})

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
                        dataset[_role][_party_id] = dataset[_role].get(_party_id, {})
                        if dsl_version == 1:
                            for _data_type, _data_location in _role_party_args[_party_index]['args']['data'].items():
                                dataset[_role][_party_id][_data_type] = '{}.{}'.format(_data_location['namespace'], _data_location['name'])
                        else:
                            for key in _role_party_args[_party_index].keys():
                                for _data_type, _data_location in _role_party_args[_party_index][key].items():
                                    dataset[_role][_party_id][key] = '{}.{}'.format(_data_location['namespace'], _data_location['name'])
        return dataset

    @classmethod
    def query_job_input_args(cls, input_data, role, party_id):
        min_partition = data_utils.get_input_data_min_partitions(input_data, role, party_id)
        return {'min_input_data_partition': min_partition}

    @classmethod
    def apply_resource(cls, job_id, role, party_id):
        return ResourceManager.apply_for_job_resource(job_id=job_id, role=role, party_id=party_id)

    @classmethod
    def return_resource(cls, job_id, role, party_id):
        return ResourceManager.return_job_resource(job_id=job_id, role=role, party_id=party_id)

    @classmethod
    def start_job(cls, job_id, role, party_id, extra_info=None):
        schedule_logger(job_id=job_id).info(f"try to start job {job_id} on {role} {party_id}")
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
        schedule_logger(job_id=job_id).info(f"start job {job_id} on {role} {party_id} successfully")

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
            ResourceManager.return_job_resource(job_id=job_info["job_id"], role=job_info["role"], party_id=job_info["party_id"])
        return update_status

    @classmethod
    def stop_job(cls, job, stop_status):
        tasks = JobSaver.query_task(job_id=job.f_job_id, role=job.f_role, party_id=job.f_party_id)
        for task in tasks:
            TaskController.stop_task(task=task, stop_status=stop_status)
        # Job status depends on the final operation result and initiator calculate

    @classmethod
    def save_pipelined_model(cls, job_id, role, party_id):
        schedule_logger(job_id).info('job {} on {} {} start to save pipeline'.format(job_id, role, party_id))
        job_dsl, job_runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=job_id, role=role,
                                                                                        party_id=party_id)
        job_parameters = job_runtime_conf.get('job_parameters', {})
        model_id = job_parameters['model_id']
        model_version = job_parameters['model_version']
        job_type = job_parameters.get('job_type', '')
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
        tracker = Tracker(job_id=job_id, role=role, party_id=party_id, model_id=model_id, model_version=model_version)
        tracker.save_pipelined_model(pipelined_buffer_object=pipeline)
        if role != 'local':
            tracker.save_machine_learning_model_info()
        schedule_logger(job_id).info('job {} on {} {} save pipeline successfully'.format(job_id, role, party_id))

    @classmethod
    def clean_job(cls, job_id, role, party_id, roles):
        schedule_logger(job_id).info('Job {} on {} {} start to clean'.format(job_id, role, party_id))
        tasks = JobSaver.query_task(job_id=job_id, role=role, party_id=party_id, only_latest=False)
        for task in tasks:
            try:
                Tracker(job_id=job_id, role=role, party_id=party_id, task_id=task.f_task_id, task_version=task.f_task_version).clean_task(roles)
                schedule_logger(job_id).info(
                    'Job {} component {} on {} {} clean done'.format(job_id, task.f_component_name, role, party_id))
            except Exception as e:
                schedule_logger(job_id).info(
                    'Job {} component {} on {} {} clean failed'.format(job_id, task.f_component_name, role, party_id))
                schedule_logger(job_id).exception(e)
        schedule_logger(job_id).info('job {} on {} {} clean done'.format(job_id, role, party_id))

    @classmethod
    def cancel_job(cls, job_id, role, party_id):
        schedule_logger(job_id).info('{} {} get cancel waiting job {} command'.format(role, party_id, job_id))
        jobs = JobSaver.query_job(job_id=job_id)
        if jobs:
            job = jobs[0]
            try:
                # You cannot delete an event directly, otherwise the status might not be updated
                status = JobQueue.update_event(job_id=job.f_job_id, initiator_role=job.f_initiator_role, initiator_party_id=job.f_initiator_party_id, job_status=JobStatus.CANCELED)
                if not status:
                    return False
            except:
                return False
            schedule_logger(job_id).info('cancel {} job successfully, job id is {}'.format(job.f_status, job.f_job_id))
            return True
        else:
            schedule_logger(job_id).warning('role {} party id {} cancel job failed, no find jod {}'.format(role, party_id, job_id))
            raise Exception('role {} party id {} cancel job failed, no find jod {}'.format(role, party_id, job_id))

