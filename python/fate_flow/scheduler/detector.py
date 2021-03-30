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
from fate_arch import storage
from fate_arch.common.base_utils import current_timestamp
from fate_flow.db.db_models import DB, Job
from fate_arch.storage import StorageSessionBase
from fate_arch.common.log import detect_logger
from fate_flow.scheduler.federated_scheduler import FederatedScheduler
from fate_flow.entity.types import JobStatus, TaskStatus, EndStatus
from fate_flow.utils import cron, job_utils
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.operation.job_saver import JobSaver
from fate_flow.manager.resource_manager import ResourceManager


class Detector(cron.Cron):
    def run_do(self):
        self.detect_running_task()
        self.detect_running_job()
        self.detect_resource_record()
        self.detect_expired_session()

    @classmethod
    def detect_running_task(cls):
        detect_logger().info('start to detect running task..')
        count = 0
        try:
            running_tasks = JobSaver.query_task(party_status=TaskStatus.RUNNING, run_on_this_party=True, run_ip=RuntimeConfig.JOB_SERVER_HOST, only_latest=False)
            stop_job_ids = set()
            for task in running_tasks:
                count += 1
                try:
                    process_exist = job_utils.check_job_process(int(task.f_run_pid))
                    if not process_exist:
                        detect_logger(job_id=task.f_job_id).info(
                                'job {} task {} {} on {} {} process {} does not exist'.format(
                                    task.f_job_id,
                                    task.f_task_id,
                                    task.f_task_version,
                                    task.f_role,
                                    task.f_party_id,
                                    task.f_run_pid))
                        stop_job_ids.add(task.f_job_id)
                except Exception as e:
                    detect_logger(job_id=task.f_job_id).exception(e)
            if stop_job_ids:
                detect_logger().info('start to stop jobs: {}'.format(stop_job_ids))
            stop_jobs = set()
            for job_id in stop_job_ids:
                jobs = JobSaver.query_job(job_id=job_id)
                if jobs:
                    stop_jobs.add(jobs[0])
            cls.request_stop_jobs(jobs=stop_jobs, stop_msg="task executor process abort", stop_status=JobStatus.CANCELED)
        except Exception as e:
            detect_logger().exception(e)
        finally:
            detect_logger().info(f"finish detect {count} running task")

    @classmethod
    def detect_running_job(cls):
        detect_logger().info('start detect running job')
        try:
            running_jobs = JobSaver.query_job(status=JobStatus.RUNNING, is_initiator=True)
            stop_jobs = set()
            for job in running_jobs:
                try:
                    if job_utils.check_job_is_timeout(job):
                        stop_jobs.add(job)
                except Exception as e:
                    detect_logger(job_id=job.f_job_id).exception(e)
            cls.request_stop_jobs(jobs=stop_jobs, stop_msg="running timeout", stop_status=JobStatus.TIMEOUT)
        except Exception as e:
            detect_logger().exception(e)
        finally:
            detect_logger().info('finish detect running job')

    @classmethod
    @DB.connection_context()
    def detect_resource_record(cls):
        detect_logger().info('start detect resource recycle')
        try:
            filter_status = EndStatus.status_list()
            filter_status.append(JobStatus.WAITING)
            jobs = Job.select().where(Job.f_resource_in_use == True, current_timestamp() - Job.f_apply_resource_time > 10 * 60 * 1000, Job.f_status << filter_status)
            stop_jobs = set()
            for job in jobs:
                if job.f_status == JobStatus.WAITING:
                    stop_jobs.add(job)
                else:
                    try:
                        detect_logger(job_id=job.f_job_id).info(f"start to return job {job.f_job_id} on {job.f_role} {job.f_party_id} resource")
                        flag = ResourceManager.return_job_resource(job_id=job.f_job_id, role=job.f_role, party_id=job.f_party_id)
                        if flag:
                            detect_logger(job_id=job.f_job_id).info(f"return job {job.f_job_id} on {job.f_role} {job.f_party_id} resource successfully")
                        else:
                            detect_logger(job_id=job.f_job_id).info(f"return job {job.f_job_id} on {job.f_role} {job.f_party_id} resource failed")
                    except Exception as e:
                        detect_logger(job_id=job.f_job_id).exception(e)
            cls.request_stop_jobs(jobs=stop_jobs, stop_msg="start timeout", stop_status=JobStatus.TIMEOUT)
        except Exception as e:
            detect_logger().exception(e)
        finally:
            detect_logger().info('finish detect resource recycle')

    @classmethod
    def detect_expired_session(cls):
        detect_logger().info('start detect expired session')
        sessions_record = StorageSessionBase.query_expired_sessions_record(ttl=5 * 60 * 60 * 1000)
        for session_record in sessions_record:
            detect_logger().info('start stop session id {}'.format(session_record.f_session_id))
            session = storage.Session.build(session_id=session_record.f_session_id, storage_engine=session_record.f_engine_name)
            session.destroy_session()
            detect_logger().info('session id {} success'.format(session_record.f_session_id))

    @classmethod
    def request_stop_jobs(cls, jobs: [Job], stop_msg, stop_status):
        if not len(jobs):
            return
        detect_logger().info(f"have {len(jobs)} should be stopped, because of {stop_msg}")
        for job in jobs:
            try:
                detect_logger(job_id=job.f_job_id).info(f"detector request start to stop job {job.f_job_id}, because of {stop_msg}")
                FederatedScheduler.request_stop_job(job=job, stop_status=stop_status)
                detect_logger(job_id=job.f_job_id).info(f"detector request stop job {job.f_job_id} successfully")
            except Exception as e:
                detect_logger(job_id=job.f_job_id).exception(e)
