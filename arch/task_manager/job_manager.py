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
from arch.task_manager.db.models import DB, JobInfo, JobQueue
from arch.task_manager.settings import logger
import datetime
import json


def save_job_info(job_id, **kwargs):
    job_info = JobInfo()
    job_info.job_id = job_id
    job_info.create_date = datetime.datetime.now()
    for k, v in kwargs.items():
        setattr(job_info, k, v)
    job_info.save(force_insert=True)


def query_job_info(job_id):
    jobs = JobInfo.select().where(JobInfo.job_id == job_id)
    return [job.to_json() for job in jobs]


def update_job_info(job_id, update_data):
    sql = JobInfo.update(**update_data).where(JobInfo.job_id == job_id)
    return sql.execute() > 0


def push_into_job_queue(job_id, config):
    job_queue = JobQueue()
    job_queue.job_id = job_id
    job_queue.status = 'waiting'
    job_queue.config = json.dumps(config)
    job_queue.create_date = datetime.datetime.now()
    job_queue.save(force_insert=True)


def get_job_from_queue(status, limit=1):
    wait_jobs = JobQueue.select().where(JobQueue.status == status).order_by(JobQueue.create_date.asc()).limit(limit)
    return [job.to_json() for job in wait_jobs]


def update_job_queue(job_id, update_data):
    sql = JobQueue.update(**update_data).where(JobQueue.job_id == job_id)
    return sql.execute() > 0


def pop_from_job_queue(job_id):
    try:
        query = JobQueue.delete().where(JobQueue.job_id == job_id)
        return query.execute() > 0
    except Exception as e:
        return False


def job_queue_size():
    return JobQueue.select().count()


def running_job_amount():
    return JobQueue.select().where(JobQueue.status == "running").count()

