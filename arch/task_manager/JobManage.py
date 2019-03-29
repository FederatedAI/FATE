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
from arch.task_manager.db.models import Job, DB
import datetime


def save_job(job_id, **kwargs):
    DB.create_tables([Job])
    job = Job()
    job.job_id = job_id
    job.create_date = datetime.datetime.now()
    for k, v in kwargs.items():
        setattr(job, k, v)
    job.save(force_insert=True)


def query_job(job_id):
    jobs = Job.select().where(Job.job_id==job_id)
    return [job.to_json() for job in jobs]


def update_job(job_id, update_data):
    query = Job.update(**update_data).where(Job.job_id==job_id)
    return query.execute()
