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
from fate_arch.common.base_utils import json_loads, current_timestamp
from fate_arch.common.log import schedule_logger
from fate_arch.common import WorkMode
from fate_arch.common import compatibility_utils
from fate_flow.db.db_models import Job
from fate_flow.scheduler.federated_scheduler import FederatedScheduler
from fate_flow.scheduler.task_scheduler import TaskScheduler
from fate_flow.operation.job_saver import JobSaver
from fate_flow.entity.constant import JobStatus, TaskStatus, EndStatus, StatusSet, SchedulingStatusCode, ResourceOperation, FederatedSchedulingStatusCode
from fate_flow.operation.job_tracker import Tracker
from fate_flow.controller.job_controller import JobController
from fate_flow.settings import FATE_BOARD_DASHBOARD_ENDPOINT, DEFAULT_TASK_PARALLELISM, DEFAULT_CORES_PER_TASK, DEFAULT_MEMORY_PER_TASK
from fate_flow.utils import detect_utils, job_utils
from fate_flow.utils.job_utils import generate_job_id, save_job_conf, get_job_log_directory, get_job_dsl_parser
from fate_flow.utils.service_utils import ServiceUtils
from fate_flow.utils import model_utils
from fate_flow.scheduler.job_queue import JobQueue
from fate_flow.utils.cron import Cron


class DAGScheduler(Cron):
    @classmethod
    def submit(cls, job_data, job_id=None):
        if not job_id:
            job_id = generate_job_id()
        schedule_logger(job_id).info('submit job, job_id {}, body {}'.format(job_id, job_data))
        job_dsl = job_data.get('job_dsl', {})
        job_runtime_conf = job_data.get('job_runtime_conf', {})
        job_parameters = job_runtime_conf['job_parameters']
        job_initiator = job_runtime_conf['initiator']

        # Compatible
        computing_engine, federation_engine, federation_mode = compatibility_utils.backend_compatibility(**job_runtime_conf["job_parameters"])
        job_parameters["computing_backend"] = job_parameters.get("computing_backend", f"DEFAULT_{computing_engine}")
        job_parameters["federation_backend"] = job_parameters.get("federation_backend", f"DEFAULT_{federation_engine}")
        job_parameters["federation_mode"] = federation_mode

        # set default parameters
        job_runtime_conf["job_parameters"]["task_parallelism"] = int(job_runtime_conf["job_parameters"].get("task_parallelism", DEFAULT_TASK_PARALLELISM))
        job_runtime_conf["job_parameters"]["cores_per_task"] = int(job_runtime_conf["job_parameters"].get("cores_per_task", DEFAULT_CORES_PER_TASK))
        job_runtime_conf["job_parameters"]["memory_per_task"] = int(job_runtime_conf["job_parameters"].get("memory_per_task", DEFAULT_MEMORY_PER_TASK))

        job_utils.check_pipeline_job_runtime_conf(job_runtime_conf)
        job_type = job_parameters.get('job_type', '')
        if job_type != 'predict':
            # generate job model info
            job_parameters['model_id'] = model_utils.gen_model_id(job_runtime_conf['role'])
            job_parameters['model_version'] = job_id
            train_runtime_conf = {}
        else:
            detect_utils.check_config(job_parameters, ['model_id', 'model_version'])
            # get inference dsl from pipeline model as job dsl
            tracker = Tracker(job_id=job_id, role=job_initiator['role'], party_id=job_initiator['party_id'],
                                  model_id=job_parameters['model_id'], model_version=job_parameters['model_version'])
            pipeline_model = tracker.get_output_model('pipeline')
            if not job_dsl:
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
        dsl_parser = get_job_dsl_parser(dsl=job_dsl,
                                        runtime_conf=job_runtime_conf,
                                        train_runtime_conf=train_runtime_conf)

        if job_parameters['work_mode'] == WorkMode.CLUSTER:
            # Save the status information of all participants in the initiator for scheduling
            for role, party_ids in job_runtime_conf["role"].items():
                for party_id in party_ids:
                    if role == job_initiator['role'] and party_id == job_initiator['party_id']:
                        continue
                    JobController.initialize_tasks(job_id, role, party_id, job_initiator, dsl_parser)

        # push into queue
        try:
            JobQueue.set_event(job_id=job_id, initiator_role=initiator_role, initiator_party_id=initiator_party_id)
        except Exception as e:
            raise Exception(f'push job into queue failed:\n{e}')

        schedule_logger(job_id).info(
            'submit job successfully, job id is {}, model id is {}'.format(job.f_job_id, job_parameters['model_id']))
        board_url = "http://{}:{}{}".format(
            ServiceUtils.get_item("fateboard", "host"),
            ServiceUtils.get_item("fateboard", "port"),
            FATE_BOARD_DASHBOARD_ENDPOINT).format(job_id, job_initiator['role'], job_initiator['party_id'])
        logs_directory = get_job_log_directory(job_id)
        return job_id, path_dict['job_dsl_path'], path_dict['job_runtime_conf_path'], logs_directory, \
               {'model_id': job_parameters['model_id'], 'model_version': job_parameters['model_version']}, board_url

    def run_do(self):
        self.schedule_waiting_jobs()
        self.schedule_running_jobs()
        self.schedule_ready_jobs()

    @classmethod
    def schedule_waiting_jobs(cls):
        schedule_logger().info("start schedule waiting jobs")
        events = JobQueue.get_event(job_status=JobStatus.WAITING)
        schedule_logger().info(f"have {len(events)} waiting jobs")
        for event in events:
            job_id, initiator_role, initiator_party_id, = event.f_job_id, event.f_initiator_role, event.f_initiator_party_id,
            update_status = JobQueue.update_event(job_id=job_id,
                                                  initiator_role=initiator_role,
                                                  initiator_party_id=initiator_party_id,
                                                  job_status=JobStatus.READY)
            if not update_status:
                schedule_logger(job_id).info(f"job {job_id} may be handled by another scheduler")
                continue
            # apply resource on all party
            jobs = job_utils.query_job(job_id=job_id, role=initiator_role, party_id=initiator_party_id)
            job = jobs[0]
            status_code, federated_response = FederatedScheduler.resource_for_job(job=job, operation_type=ResourceOperation.APPLY)
            if status_code == FederatedSchedulingStatusCode.SUCCESS:
                cls.start_job(job_id=job_id, initiator_role=initiator_role, initiator_party_id=initiator_party_id)
                update_status = JobQueue.update_event(job_id=job_id,
                                                      initiator_role=initiator_role,
                                                      initiator_party_id=initiator_party_id,
                                                      job_status=JobStatus.RUNNING)
                if not update_status:
                    pass
            else:
                # rollback resource
                rollback_party = {}
                failed_party = {}
                for dest_role in federated_response.keys():
                    for dest_party_id in federated_response[dest_role].keys():
                        retcode = federated_response[dest_role][dest_party_id]["retcode"]
                        if retcode == 0:
                            rollback_party[dest_role] = rollback_party.get(dest_role, [])
                            rollback_party[dest_role].append(dest_party_id)
                        else:
                            failed_party[dest_role] = failed_party.get(dest_role, [])
                            failed_party[dest_role].append(dest_party_id)
                schedule_logger(job_id).info("job {} apply resource failed on {}, rollback {}".format(
                    job_id,
                    ",".join([",".join([f"{_r}:{_p}" for _p in _ps]) for _r, _ps in failed_party.items()]),
                    ",".join([",".join([f"{_r}:{_p}" for _p in _ps]) for _r, _ps in rollback_party.items()]),
                ))
                if rollback_party:
                    status_code, federated_response = FederatedScheduler.resource_for_job(job=job, operation_type=ResourceOperation.RETURN, specific_dest=rollback_party)
                    if status_code != FederatedSchedulingStatusCode.SUCCESS:
                        schedule_logger(job_id).info(f"job {job_id} return resource failed:\n{federated_response}")
                else:
                    schedule_logger(job_id).info(f"job {job_id} no party should be rollback resource")
                update_status = JobQueue.update_event(job_id=job_id,
                                                      initiator_role=initiator_role,
                                                      initiator_party_id=initiator_party_id,
                                                      job_status=JobStatus.WAITING)
                schedule_logger(job_id).info(f"update job {job_id} status to waiting {update_status}")
        schedule_logger().info("schedule waiting jobs finished")

    @classmethod
    def schedule_running_jobs(cls):
        schedule_logger().info("start schedule running jobs")
        events = JobQueue.get_event(job_status=JobStatus.RUNNING)
        schedule_logger().info(f"have {len(events)} running jobs")
        for event in events:
            try:
                jobs = JobSaver.query_job(job_id=event.f_job_id, initiator_role=event.f_initiator_role, initiator_party_id=event.f_initiator_party_id, is_initiator=1, status=JobStatus.RUNNING)
                if jobs:
                    cls.schedule_job(job=jobs[0])
            except Exception as e:
                schedule_logger(event.f_job_id).error(f"schedule job {event.f_job_id} failed:\n{e}")
        schedule_logger().info("schedule running jobs finished")

    @classmethod
    def schedule_ready_jobs(cls):
        # some ready job exit before start
        schedule_logger().info("start schedule ready jobs")
        events = JobQueue.get_event(job_status=JobStatus.READY)
        schedule_logger().info(f"have {len(events)} ready jobs")
        for event in events:
            try:
                job_id, initiator_role, initiator_party_id, = event.f_job_id, event.f_initiator_role, event.f_initiator_party_id,
                update_status = JobQueue.update_event(job_id=job_id,
                                                      initiator_role=initiator_role,
                                                      initiator_party_id=initiator_party_id,
                                                      job_status=JobStatus.WAITING,
                                                      ttl=5*60*1000)
                schedule_logger(job_id).info(f"update job {job_id} ready status to waiting {update_status}")
            except Exception as e:
                schedule_logger(event.f_job_id).error(f"schedule ready job {event.f_job_id} failed:\n{e}")
        schedule_logger().info("schedule ready jobs finished")

    @classmethod
    def start_job(cls, job_id, initiator_role, initiator_party_id):
        schedule_logger(job_id=job_id).info("Try to start job {} on initiator {} {}".format(job_id, initiator_role, initiator_party_id))
        job_info = {}
        job_info["job_id"] = job_id
        job_info["role"] = initiator_role
        job_info["party_id"] = initiator_party_id
        job_info["status"] = JobStatus.RUNNING
        job_info["party_status"] = JobStatus.RUNNING
        job_info["start_time"] = current_timestamp()
        job_info["tag"] = 'end_waiting'
        jobs = job_utils.query_job(job_id=job_id, role=initiator_role, party_id=initiator_party_id)
        if jobs:
            job = jobs[0]
            FederatedScheduler.start_job(job=job)
            schedule_logger(job_id=job_id).info("start job {} on initiator {} {}".format(job_id, initiator_role, initiator_party_id))
        else:
            schedule_logger(job_id=job_id).error("can not found job {} on initiator {} {}".format(job_id, initiator_role, initiator_party_id))

    @classmethod
    def schedule_job(cls, job):
        schedule_logger(job_id=job.f_job_id).info("scheduling job {}".format(job.f_job_id))
        dsl_parser = job_utils.get_job_dsl_parser(dsl=job.f_dsl,
                                                  runtime_conf=job.f_runtime_conf,
                                                  train_runtime_conf=job.f_train_runtime_conf)
        task_scheduling_status_code, tasks = TaskScheduler.schedule(job=job, dsl_parser=dsl_parser)
        tasks_status = [task.f_status for task in tasks]
        new_job_status = cls.calculate_job_status(task_scheduling_status_code=task_scheduling_status_code, tasks_status=tasks_status)
        total, finished_count = cls.calculate_job_progress(tasks_status=tasks_status)
        new_progress = float(finished_count) / total * 100
        schedule_logger(job_id=job.f_job_id).info("Job {} status is {}, calculate by task status list: {}".format(job.f_job_id, new_job_status, tasks_status))
        if new_job_status != job.f_status or new_progress != job.f_progress:
            # Make sure to update separately, because these two fields update with anti-weight logic
            if new_progress != job.f_progress:
                job.f_progress = new_progress
                FederatedScheduler.sync_job(job=job, update_fields=["progress"])
                cls.update_job_on_initiator(initiator_job=job, update_fields=["progress"])
            if new_job_status != job.f_status:
                job.f_status = new_job_status
                if EndStatus.contains(job.f_status):
                    FederatedScheduler.save_pipelined_model(job=job)
                FederatedScheduler.sync_job(job=job, update_fields=["status"])
                cls.update_job_on_initiator(initiator_job=job, update_fields=["status"])
        if EndStatus.contains(job.f_status):
            cls.finish(job=job, end_status=job.f_status)
        schedule_logger(job_id=job.f_job_id).info("finish scheduling job {}".format(job.f_job_id))

    @classmethod
    def update_job_on_initiator(cls, initiator_job: Job, update_fields: list):
        jobs = JobSaver.query_job(job_id=initiator_job.f_job_id)
        if not jobs:
            raise Exception("Failed to update job status on initiator")
        job_info = initiator_job.to_human_model_dict(only_primary_with=update_fields)
        for field in update_fields:
            job_info[field] = getattr(initiator_job, "f_%s" % field)
        for job in jobs:
            job_info["role"] = job.f_role
            job_info["party_id"] = job.f_party_id
            JobSaver.update_job(job_info=job_info)

    @classmethod
    def calculate_job_status(cls, task_scheduling_status_code, tasks_status):
        # 1. all waiting
        # 2. have running
        # 3. waiting + end status
        # 4. all end status and difference
        # 5. all the same end status
        tmp_status_set = set(tasks_status)
        if len(tmp_status_set) == 1:
            # 1 and 5
            return tmp_status_set.pop()
        else:
            if TaskStatus.RUNNING in tmp_status_set:
                # 2
                return JobStatus.RUNNING
            if TaskStatus.WAITING in tmp_status_set:
                # 3
                if task_scheduling_status_code == SchedulingStatusCode.HAVE_NEXT:
                    return JobStatus.RUNNING
                else:
                    pass
            # 3 with no next and 4
            for status in sorted(EndStatus.status_list(), key=lambda s: StatusSet.get_level(status=s), reverse=True):
                if status == TaskStatus.COMPLETE:
                    continue
                elif status in tmp_status_set:
                    return status

            raise Exception("Calculate job status failed: {}".format(tasks_status))

    @classmethod
    def calculate_job_progress(cls, tasks_status):
        total = 0
        finished_count = 0
        for task_status in tasks_status:
            total += 1
            if EndStatus.contains(task_status):
                finished_count += 1
        return total, finished_count

    @classmethod
    def finish(cls, job, end_status):
        schedule_logger(job_id=job.f_job_id).info("Job {} finished with {}, do something...".format(job.f_job_id, end_status))
        FederatedScheduler.stop_job(job=job, stop_status=end_status)
        FederatedScheduler.clean_job(job=job)
        JobQueue.delete_event(job_id=job.f_job_id, initiator_role=job.f_initiator_role, initiator_party_id=job.f_initiator_party_id)
        schedule_logger(job_id=job.f_job_id).info("Job {} finished with {}, done".format(job.f_job_id, end_status))
