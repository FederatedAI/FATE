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
from arch.api.utils import dtable_utils
from arch.api.utils.core_utils import json_loads
from arch.api.utils.log_utils import schedule_logger
from fate_flow.db.db_models import Job
from fate_flow.scheduler.federated_scheduler import FederatedScheduler
from fate_flow.scheduler.task_scheduler import TaskScheduler
from fate_flow.operation.job_saver import JobSaver
from fate_flow.entity.constant import JobStatus, TaskSetStatus, EndStatus, InterruptStatus
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.operation.job_tracker import Tracker
from fate_flow.controller.job_controller import JobController
from fate_flow.settings import FATE_BOARD_DASHBOARD_ENDPOINT
from fate_flow.utils import detect_utils, job_utils
from fate_flow.utils.job_utils import generate_job_id, save_job_conf, get_job_log_directory
from fate_flow.utils.service_utils import ServiceUtils
from fate_flow.scheduler.status_engine import StatusEngine


class DAGScheduler(object):
    @staticmethod
    def submit(job_data, job_id=None):
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
            tracker = Tracker(job_id=job_id, role=job_initiator['role'], party_id=job_initiator['party_id'],
                                  model_id=job_parameters['model_id'], model_version=job_parameters['model_version'])
            pipeline_model = tracker.get_output_model('pipeline')
            job_dsl = json_loads(pipeline_model['Pipeline'].inference_dsl)
            train_runtime_conf = json_loads(pipeline_model['Pipeline'].train_runtime_conf)
        path_dict = save_job_conf(job_id=job_id,
                                  job_dsl=job_dsl,
                                  job_runtime_conf=job_runtime_conf,
                                  train_runtime_conf=train_runtime_conf,
                                  pipeline_dsl=None)

        job = Job()
        job.f_job_id = job_id
        job.f_dsl = job_dsl
        job.f_runtime_conf = job_runtime_conf
        job.f_train_runtime_conf = train_runtime_conf
        job.f_roles = job_runtime_conf['role']
        job.f_work_mode = job_parameters['work_mode']
        job.f_initiator_role = job_initiator['role']
        job.f_initiator_party_id = job_initiator['party_id']

        initiator_role = job_initiator['role']
        initiator_party_id = job_initiator['party_id']
        if initiator_party_id not in job_runtime_conf['role'][initiator_role]:
            schedule_logger(job_id).info("initiator party id error:{}".format(initiator_party_id))
            raise Exception("initiator party id error {}".format(initiator_party_id))

        FederatedScheduler.create_job(job=job)

        # Save the state information of all participants in the initiator for scheduling
        job_info = job.to_dict_info()
        for role, party_ids in job_runtime_conf["role"].items():
            for party_id in party_ids:
                if role == job_initiator['role'] and party_id == job_initiator['party_id']:
                    continue
                JobController.create_job(job_id=job_id, role=role, party_id=party_id, job_info=job_info, init_tracker=False)

        # push into queue
        job_event = job_utils.job_event(job_id, initiator_role, initiator_party_id)
        try:
            RuntimeConfig.JOB_QUEUE.put_event(job_event)
        except Exception as e:
            raise Exception('push job into queue failed')

        schedule_logger(job_id).info(
            'submit job successfully, job id is {}, model id is {}'.format(job.f_job_id, job_parameters['model_id']))
        board_url = "http://{}:{}{}".format(
            ServiceUtils.get_item("fateboard", "host"),
            ServiceUtils.get_item("fateboard", "port"),
            FATE_BOARD_DASHBOARD_ENDPOINT).format(job_id, job_initiator['role'], job_initiator['party_id'])
        logs_directory = get_job_log_directory(job_id)
        return job_id, path_dict['job_dsl_path'], path_dict['job_runtime_conf_path'], logs_directory, \
               {'model_id': job_parameters['model_id'], 'model_version': job_parameters['model_version']}, board_url

    @staticmethod
    def start(job_id, initiator_role, initiator_party_id):
        schedule_logger(job_id=job_id).info("Try to start job {} on initiator {} {}".format(job_id, initiator_role, initiator_party_id))
        job_info = {}
        job_info["job_id"] = job_id
        job_info["role"] = initiator_role
        job_info["party_id"] = initiator_party_id
        job_info["status"] = JobStatus.RUNNING
        job_info["party_status"] = JobStatus.RUNNING
        job_info["tag"] = 'end_waiting'
        update_status = JobSaver.update_job(job_info=job_info)
        if update_status:
            jobs = job_utils.query_job(job_id=job_id, role=initiator_role, party_id=initiator_party_id)
            job = jobs[0]
            FederatedScheduler.start_job(job=job)
            schedule_logger(job_id=job_id).info("Start job {} on initiator {} {}".format(job_id, initiator_role, initiator_party_id))
        else:
            schedule_logger(job_id=job_id).info("Job {} start on another scheduler".format(job_id))

    @classmethod
    def schedule(cls, job):
        schedule_logger(job_id=job.f_job_id).info("Schedule job {}".format(job.f_job_id))
        task_sets = JobSaver.get_top_task_set(job_id=job.f_job_id, role=job.f_role, party_id=job.f_party_id)
        if not task_sets:
            # Maybe for some reason the initialization failed
            return
        for task_set in task_sets:
            if task_set.f_status == TaskSetStatus.WAITING:
                schedule_logger(job_id=task_set.f_job_id).info("Try to start job {} task set {} on {} {}".format(task_set.f_job_id, task_set.f_task_set_id, task_set.f_role, task_set.f_party_id))
                task_set.f_status = TaskSetStatus.RUNNING
                update_status = JobSaver.update_task_set(task_set_info=task_set.to_dict_info(only_primary_with=["job_id", "status"]))
                if not update_status:
                    # another scheduler
                    schedule_logger(job_id=job.f_job_id).info("Job {} task set {} start on another scheduler".format(task_set.f_job_id, task_set.f_task_set_id))
                    break
                schedule_logger(job_id=job.f_job_id).info("Start job {} task set {} on {} {}".format(task_set.f_job_id, task_set.f_task_set_id, task_set.f_role, task_set.f_party_id))
                TaskScheduler.schedule(job=job, task_set=task_set)
                break
            elif task_set.f_status == TaskSetStatus.RUNNING:
                # TODO: Determine whether it has timed out
                schedule_logger(job_id=job.f_job_id).info("Job {} task set {} is running".format(task_set.f_job_id, task_set.f_task_set_id))
                TaskScheduler.schedule(job=job, task_set=task_set)
                break
            elif InterruptStatus.is_interrupt_status(task_set.f_status):
                schedule_logger(job_id=job.f_job_id).info("Job {} task set {} is {}, job exit".format(task_set.f_job_id, task_set.f_task_set_id, task_set.f_status))
                break
            elif task_set.f_status == TaskSetStatus.COMPLETE:
                continue
            else:
                raise Exception("Can not scheduling job {}".format(job.f_job_id))
        new_job_status = StatusEngine.vertical_convergence([task_set.f_status for task_set in task_sets])
        schedule_logger(job_id=job.f_job_id).info("Job {} status is {}".format(job.f_job_id, new_job_status))
        if new_job_status != job.f_status:
            job.f_status = new_job_status
            FederatedScheduler.sync_job(job=job, update_fields=["status"])
            cls.update_job_on_initiator(initiator_job_template=job, update_fields=["status"])
        if EndStatus.is_end_status(job.f_status):
            cls.finish(job=job, end_status=job.f_status)

    @classmethod
    def update_job_on_initiator(cls, initiator_job_template: Job, update_fields: list):
        jobs = JobSaver.query_job(job_id=initiator_job_template.f_job_id)
        if not jobs:
            raise Exception("Failed to update job status on initiator")
        job_info = initiator_job_template.to_dict_info(only_primary_with=update_fields)
        for field in update_fields:
            job_info[field] = getattr(initiator_job_template, "f_%s" % field)
        for job in jobs:
            job_info["role"] = job.f_role
            job_info["party_id"] = job.f_party_id
            JobSaver.update_job(job_info=job_info)

    @staticmethod
    def finish(job, end_status):
        schedule_logger(job_id=job.f_job_id).info("Job {} finished, do something...".format(job.f_job_id))
        FederatedScheduler.save_pipelined_model(job=job)
        FederatedScheduler.stop_job(job=job, stop_status=end_status)
        FederatedScheduler.clean_job(job=job)
        schedule_logger(job_id=job.f_job_id).info("Job {} finished, done".format(job.f_job_id))

    @staticmethod
    def clean_queue():
        schedule_logger().info('get clean queue command')
        jobs = job_utils.query_job(is_initiator=1, status=JobStatus.WAITING)
        if jobs:
            for job in jobs:
                schedule_logger(job.f_job_id).info(
                    'start send {} job {} command success'.format(JobStatus.CANCELED, job.f_job_id))
                job_info = {'f_job_id': job.f_job_id, 'f_status': JobStatus.CANCELED}
                roles = json_loads(job.f_roles)
                job_work_mode = job.f_work_mode
                initiator_party_id = job.f_party_id

                TaskScheduler.sync_job_status(job_id=job.f_job_id, roles=roles, initiator_party_id=initiator_party_id,
                                              initiator_role=job.f_role,
                                              work_mode=job_work_mode,
                                              job_info=job_info)
                job_runtime_conf = job.f_runtime_conf
                event = job_utils.job_event(job.f_job_id,
                                            job_runtime_conf['initiator']['role'],
                                            job_runtime_conf['initiator']['party_id'])
                try:
                    RuntimeConfig.JOB_QUEUE.del_event(event)
                    schedule_logger(job.f_job_id).info(
                        'send {} job {} command success'.format(JobStatus.CANCELED, job.f_job_id))
                except Exception as e:
                    schedule_logger(job.f_job_id).error(e)
        else:
            raise Exception('There are no jobs in the queue')
