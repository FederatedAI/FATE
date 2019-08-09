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
from arch.api.proto import pipeline_pb2
from arch.api.utils import dtable_utils
from arch.api.utils.core import current_timestamp, json_dumps, json_loads, get_lan_ip
from fate_flow.db.db_models import Job
from fate_flow.driver.task_executor import TaskExecutor
from fate_flow.driver.task_scheduler import TaskScheduler
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.manager.tracking import Tracking
from fate_flow.settings import schedule_logger
from fate_flow.utils import job_utils
from fate_flow.utils.job_utils import generate_job_id, save_job_conf, get_job_dsl_parser


class JobController(object):
    task_executor_pool = None

    @staticmethod
    def init():
        pass

    @staticmethod
    def submit_job(job_data):
        job_id = generate_job_id()
        schedule_logger.info('submit job, job_id {}, body {}'.format(job_id, job_data))
        job_runtime_conf = job_data.get('job_runtime_conf', {})
        job_dsl = job_data.get('job_dsl', {})
        job_utils.check_pipeline_job_runtime_conf(job_runtime_conf)
        job_parameters = job_runtime_conf['job_parameters']
        job_type = job_parameters.get('type', '')
        if job_type != 'predict':
            job_parameters['model_id'] = '#'.join([dtable_utils.all_party_key(job_runtime_conf['role']), 'model'])
            job_parameters['model_version'] = job_id
        else:
            job_utils.check_config(job_parameters, ['model_id', 'model_version'])
        job_dsl_path, job_runtime_conf_path = save_job_conf(job_id=job_id,
                                                            job_dsl=job_dsl,
                                                            job_runtime_conf=job_runtime_conf)
        job_initiator = job_runtime_conf['initiator']

        job = Job()
        job.f_job_id = job_id
        job.f_roles = json_dumps(job_runtime_conf['role'])
        job.f_work_mode = job_parameters['work_mode']
        job.f_initiator_party_id = job_initiator['party_id']
        job.f_dsl = json_dumps(job_dsl)
        job.f_runtime_conf = json_dumps(job_runtime_conf)
        job.f_run_ip = get_lan_ip()
        job.f_status = 'waiting'
        job.f_progress = 0
        job.f_create_time = current_timestamp()

        # save job info
        TaskScheduler.distribute_job(job=job, roles=job_runtime_conf['role'], job_initiator=job_initiator)

        # generate model info
        model_info = JobController.gen_model_info(job_runtime_conf['role'], job_parameters['model_id'],
                                                  job_parameters['model_version'])
        # push into queue
        RuntimeConfig.JOB_QUEUE.put_event({
            'job_id': job_id,
            "job_dsl_path": job_dsl_path,
            "job_runtime_conf_path": job_runtime_conf_path
        }
        )
        schedule_logger.info(
            'submit job successfully, job id is {}, model id is {}'.format(job.f_job_id, job_parameters['model_id']))
        return job_id, job_dsl_path, job_runtime_conf_path, model_info

    @staticmethod
    def gen_model_info(roles, model_id, model_version):
        model_info = {'model_id': model_id, 'model_version': model_version}
        for _role, role_partys in roles.items():
            model_info[_role] = {}
            for _party_id in role_partys:
                model_info[_role][_party_id] = Tracking.gen_party_model_id(model_id, role=_role, party_id=_party_id)
        return model_info

    @staticmethod
    def kill_job(job_id, role, party_id, job_initiator):
        schedule_logger.info('{} {} get kill job {} command'.format(role, party_id, job_id))
        tasks = job_utils.query_task(job_id=job_id, role=role, party_id=party_id)
        for task in tasks:
            kill_status = False
            try:
                kill_status = job_utils.kill_process(int(task.f_run_pid))
            except Exception as e:
                schedule_logger.exception(e)
            finally:
                schedule_logger.info(
                    'job {} component {} on {} {} process {} kill {}'.format(job_id, task.f_component_name, task.f_role,
                                                                             task.f_party_id, task.f_run_pid,
                                                                             'success' if kill_status else 'failed'))
            if task.f_status != 'success':
                task.f_status = 'failed'
            TaskExecutor.sync_task_status(job_id=job_id, component_name=task.f_component_name, task_id=task.f_task_id,
                                          role=role,
                                          party_id=party_id, initiator_party_id=job_initiator.get('party_id', None),
                                          task_info=task.to_json())

    @staticmethod
    def update_task_status(job_id, component_name, task_id, role, party_id, task_info):
        tracker = Tracking(job_id=job_id, role=role, party_id=party_id, component_name=component_name, task_id=task_id)
        tracker.save_task(role=role, party_id=party_id, task_info=task_info)
        schedule_logger.info(
            'job {} component {} {} {} status {}'.format(job_id, component_name, role, party_id,
                                                         task_info.get('f_status', '')))

    @staticmethod
    def update_job_status(job_id, role, party_id, job_info, create=False):
        job_tracker = Tracking(job_id=job_id, role=role, party_id=party_id)
        if create:
            dsl = json_loads(job_info['f_dsl'])
            runtime_conf = json_loads(job_info['f_runtime_conf'])
            save_job_conf(job_id=job_id,
                          job_dsl=dsl,
                          job_runtime_conf=runtime_conf)
            roles = json_loads(job_info['f_roles'])
            partner = {}
            show_role = {}
            is_initiator = job_info.get('f_is_initiator', 0)
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

            dag = get_job_dsl_parser(dsl=dsl,
                                     runtime_conf=runtime_conf)
            job_args = dag.get_args_input()
            dataset = {}
            for _role, _role_party_args in job_args.items():
                if is_initiator or _role == role:
                    for _party_index in range(len(_role_party_args)):
                        _party_id = roles[_role][_party_index]
                        if is_initiator or _party_id == party_id:
                            dataset[_role] = dataset.get(_role, {})
                            dataset[_role][_party_id] = dataset[_role].get(_party_id, {})
                            for _data_type, _data_location in _role_party_args[_party_index]['args']['data'].items():
                                dataset[_role][_party_id][_data_type] = '{}.{}'.format(_data_location['namespace'],
                                                                                       _data_location['name'])
            job_tracker.log_job_view({'partner': partner, 'dataset': dataset, 'roles': show_role})
        job_tracker.save_job_info(role=role, party_id=party_id, job_info=job_info, create=create)

    @staticmethod
    def save_pipeline(job_id, role, party_id, model_id, model_version):
        job_dsl, job_runtime_conf = job_utils.get_job_configuration(job_id=job_id, role=role, party_id=party_id)
        job_parameters = job_runtime_conf.get('job_parameters', {})
        job_type = job_parameters.get('type', '')
        if job_type == 'predict':
            return
        dag = job_utils.get_job_dsl_parser(dsl=job_dsl,
                                           runtime_conf=job_runtime_conf)
        predict_dsl = dag.get_predict_dsl(role=role)
        pipeline = pipeline_pb2.Pipeline()
        pipeline.inference_dsl = json_dumps(predict_dsl, byte=True)
        pipeline.train_dsl = json_dumps(job_dsl, byte=True)
        pipeline.train_runtime_conf = json_dumps(job_runtime_conf, byte=True)
        job_tracker = Tracking(job_id=job_id, role=role, party_id=party_id, model_id=model_id,
                               model_version=model_version)
        job_tracker.save_output_model({'Pipeline': pipeline}, 'pipeline')

    @staticmethod
    def clean_job(job_id, role, party_id):
        schedule_logger.info('job {} on {} {} start to clean'.format(job_id, role, party_id))
        tasks = job_utils.query_task(job_id=job_id, role=role, party_id=party_id)
        for task in tasks:
            try:
                Tracking(job_id=job_id, role=role, party_id=party_id, task_id=task.f_task_id).clean_task()
                schedule_logger.info(
                    'job {} component {} on {} {} clean done'.format(job_id, task.f_component_name, role, party_id))
            except Exception as e:
                schedule_logger.info(
                    'job {} component {} on {} {} clean failed'.format(job_id, task.f_component_name, role, party_id))
                schedule_logger.exception(e)
        schedule_logger.info('job {} on {} {} clean done'.format(job_id, role, party_id))
