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
from arch.api.utils import dtable_utils
from arch.api.utils.core_utils import current_timestamp, json_dumps, json_loads
from arch.api.utils.log_utils import schedule_logger
from fate_flow.db.db_models import Job
from fate_flow.driver.task_executor import TaskExecutor
from fate_flow.driver.task_scheduler import TaskScheduler
from fate_flow.entity.constant_config import JobStatus, TaskStatus
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.manager.tracking_manager import Tracking
from fate_flow.settings import BOARD_DASHBOARD_URL, USE_AUTHENTICATION
from fate_flow.utils import detect_utils, job_utils
from fate_flow.utils.job_utils import generate_job_id, save_job_conf, get_job_dsl_parser, get_job_log_directory


class JobController(object):
    task_executor_pool = None

    @staticmethod
    def init():
        pass

    @staticmethod
    def submit_job(job_data, job_id=None):
        if not job_id:
            job_id = generate_job_id()
        schedule_logger(job_id).info('submit job, job_id {}, body {}'.format(job_id, job_data))
        job_dsl = job_data.get('job_dsl', {})
        job_runtime_conf = job_data.get('job_runtime_conf', {})
        job_utils.check_pipeline_job_runtime_conf(job_runtime_conf)
        job_parameters = job_runtime_conf['job_parameters']
        job_initiator = job_runtime_conf['initiator']
        job_type = job_parameters.get('job_type', '')
        if job_type != 'predict':
            # generate job model info
            job_parameters['model_id'] = '#'.join([dtable_utils.all_party_key(job_runtime_conf['role']), 'model'])
            job_parameters['model_version'] = job_id
            train_runtime_conf = {}
        else:
            detect_utils.check_config(job_parameters, ['model_id', 'model_version'])
            # get inference dsl from pipeline model as job dsl
            job_tracker = Tracking(job_id=job_id, role=job_initiator['role'], party_id=job_initiator['party_id'],
                                   model_id=job_parameters['model_id'], model_version=job_parameters['model_version'])
            pipeline_model = job_tracker.get_output_model('pipeline')
            job_dsl = json_loads(pipeline_model['Pipeline'].inference_dsl)
            train_runtime_conf = json_loads(pipeline_model['Pipeline'].train_runtime_conf)
        path_dict = save_job_conf(job_id=job_id,
                                  job_dsl=job_dsl,
                                  job_runtime_conf=job_runtime_conf,
                                  train_runtime_conf=train_runtime_conf,
                                  pipeline_dsl=None)

        job = Job()
        job.f_job_id = job_id
        job.f_roles = json_dumps(job_runtime_conf['role'])
        job.f_work_mode = job_parameters['work_mode']
        job.f_initiator_party_id = job_initiator['party_id']
        job.f_dsl = json_dumps(job_dsl)
        job.f_runtime_conf = json_dumps(job_runtime_conf)
        job.f_train_runtime_conf = json_dumps(train_runtime_conf)
        job.f_run_ip = ''
        job.f_status = JobStatus.WAITING
        job.f_progress = 0
        job.f_create_time = current_timestamp()

        initiator_role = job_initiator['role']
        initiator_party_id = job_initiator['party_id']
        if initiator_party_id not in job_runtime_conf['role'][initiator_role]:
            schedule_logger(job_id).info("initiator party id error:{}".format(initiator_party_id))
            raise Exception("initiator party id error {}".format(initiator_party_id))

        get_job_dsl_parser(dsl=job_dsl,
                           runtime_conf=job_runtime_conf,
                           train_runtime_conf=train_runtime_conf)

        TaskScheduler.distribute_job(job=job, roles=job_runtime_conf['role'], job_initiator=job_initiator)

        # push into queue
        job_event = job_utils.job_event(job_id, initiator_role,  initiator_party_id)
        try:
            RuntimeConfig.JOB_QUEUE.put_event(job_event)
        except Exception as e:
            raise Exception('push job into queue failed')

        schedule_logger(job_id).info(
            'submit job successfully, job id is {}, model id is {}'.format(job.f_job_id, job_parameters['model_id']))
        board_url = BOARD_DASHBOARD_URL.format(job_id, job_initiator['role'], job_initiator['party_id'])
        logs_directory = get_job_log_directory(job_id)
        return job_id, path_dict['job_dsl_path'], path_dict['job_runtime_conf_path'], logs_directory, \
               {'model_id': job_parameters['model_id'],'model_version': job_parameters['model_version']}, board_url

    @staticmethod
    def kill_job(job_id, role, party_id, job_initiator, timeout=False, component_name=''):
        schedule_logger(job_id).info('{} {} get kill job {} {} command'.format(role, party_id, job_id, component_name))
        task_info = job_utils.get_task_info(job_id, role, party_id, component_name)
        tasks = job_utils.query_task(**task_info)
        job = job_utils.query_job(job_id=job_id)
        for task in tasks:
            kill_status = False
            try:
                # task clean up
                runtime_conf = json_loads(job[0].f_runtime_conf)
                roles = ','.join(runtime_conf['role'].keys())
                party_ids = ','.join([','.join([str(j) for j in i]) for i in runtime_conf['role'].values()])
                # Tracking(job_id=job_id, role=role, party_id=party_id, task_id=task.f_task_id).clean_task(roles, party_ids)
                # stop task
                kill_status = job_utils.kill_task_executor_process(task)
                # session stop
                job_utils.start_session_stop(task)
            except Exception as e:
                schedule_logger(job_id).exception(e)
            finally:
                schedule_logger(job_id).info(
                    'job {} component {} on {} {} process {} kill {}'.format(job_id, task.f_component_name, task.f_role,
                                                                             task.f_party_id, task.f_run_pid,
                                                                             'success' if kill_status else 'failed'))
            status = TaskStatus.FAILED if not timeout else TaskStatus.TIMEOUT

            if task.f_status != TaskStatus.COMPLETE:
                task.f_status = status
            try:
                TaskExecutor.sync_task_status(job_id=job_id, component_name=task.f_component_name, task_id=task.f_task_id,
                                              role=role,
                                              party_id=party_id, initiator_party_id=job_initiator.get('party_id', None),
                                              task_info=task.to_json(), initiator_role=job_initiator.get('role', None))
            except Exception as e:
                schedule_logger(job_id).exception(e)

    @staticmethod
    def update_task_status(job_id, component_name, task_id, role, party_id, task_info):
        tracker = Tracking(job_id=job_id, role=role, party_id=party_id, component_name=component_name, task_id=task_id)
        tracker.save_task(role=role, party_id=party_id, task_info=task_info)
        schedule_logger(job_id).info(
            'job {} component {} {} {} status {}'.format(job_id, component_name, role, party_id,
                                                         task_info.get('f_status', '')))

    @staticmethod
    def update_job_status(job_id, role, party_id, job_info, create=False):
        job_info['f_run_ip'] = RuntimeConfig.JOB_SERVER_HOST
        if create:
            dsl = json_loads(job_info['f_dsl'])
            runtime_conf = json_loads(job_info['f_runtime_conf'])
            train_runtime_conf = json_loads(job_info['f_train_runtime_conf'])
            if USE_AUTHENTICATION:
                authentication_check(src_role=job_info.get('src_role', None), src_party_id=job_info.get('src_party_id', None),
                                     dsl=dsl, runtime_conf=runtime_conf, role=role, party_id=party_id)
            save_job_conf(job_id=job_id,
                          job_dsl=dsl,
                          job_runtime_conf=runtime_conf,
                          train_runtime_conf=train_runtime_conf,
                          pipeline_dsl=None)

            job_parameters = runtime_conf['job_parameters']
            job_tracker = Tracking(job_id=job_id, role=role, party_id=party_id,
                                   model_id=job_parameters["model_id"],
                                   model_version=job_parameters["model_version"])
            job_tracker.job_quantity_constraint()
            if job_parameters.get("job_type", "") != "predict":
                job_tracker.init_pipelined_model()
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
                                     runtime_conf=runtime_conf,
                                     train_runtime_conf=train_runtime_conf)
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
        else:
            job_tracker = Tracking(job_id=job_id, role=role, party_id=party_id)
        job_tracker.save_job_info(role=role, party_id=party_id, job_info=job_info, create=create)

    @staticmethod
    def save_pipeline(job_id, role, party_id, model_id, model_version):
        schedule_logger(job_id).info('job {} on {} {} start to save pipeline'.format(job_id, role, party_id))
        job_dsl, job_runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=job_id, role=role,
                                                                                        party_id=party_id)
        job_parameters = job_runtime_conf.get('job_parameters', {})
        job_type = job_parameters.get('job_type', '')
        if job_type == 'predict':
            return
        dag = job_utils.get_job_dsl_parser(dsl=job_dsl,
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
        job_tracker = Tracking(job_id=job_id, role=role, party_id=party_id, model_id=model_id,
                               model_version=model_version)
        job_tracker.save_pipeline(pipelined_buffer_object=pipeline)
        schedule_logger(job_id).info('job {} on {} {} save pipeline successfully'.format(job_id, role, party_id))

    @staticmethod
    def clean_job(job_id, role, party_id, roles, party_ids):
        schedule_logger(job_id).info('job {} on {} {} start to clean'.format(job_id, role, party_id))
        tasks = job_utils.query_task(job_id=job_id, role=role, party_id=party_id)
        for task in tasks:
            try:
                Tracking(job_id=job_id, role=role, party_id=party_id, task_id=task.f_task_id).clean_task(roles, party_ids)
                schedule_logger(job_id).info(
                    'job {} component {} on {} {} clean done'.format(job_id, task.f_component_name, role, party_id))
            except Exception as e:
                schedule_logger(job_id).info(
                    'job {} component {} on {} {} clean failed'.format(job_id, task.f_component_name, role, party_id))
                schedule_logger(job_id).exception(e)
        schedule_logger(job_id).info('job {} on {} {} clean done'.format(job_id, role, party_id))

    @staticmethod
    def cancel_job(job_id, role, party_id, job_initiator):
        schedule_logger(job_id).info('{} {} get cancel waiting job {} command'.format(role, party_id, job_id))
        jobs = job_utils.query_job(job_id=job_id, is_initiator=1)
        if jobs:
            job = jobs[0]
            job_runtime_conf = json_loads(job.f_runtime_conf)
            event = job_utils.job_event(job.f_job_id,
                                        job_runtime_conf['initiator']['role'],
                                        job_runtime_conf['initiator']['party_id'])
            try:
                RuntimeConfig.JOB_QUEUE.del_event(event)
            except:
                return False
            schedule_logger(job_id).info('cancel waiting job successfully, job id is {}'.format(job.f_job_id))
            return True
        else:
            jobs = job_utils.query_job(job_id=job_id)
            if jobs:
                raise Exception(
                    'role {} party id {} cancel waiting job {} failed, not is initiator'.format(role, party_id, job_id))
            raise Exception('role {} party id {} cancel waiting job failed, no find jod {}'.format(role, party_id, job_id))





