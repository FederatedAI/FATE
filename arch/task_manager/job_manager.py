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
from arch.api.utils import file_utils
import subprocess
import os
import uuid
from arch.task_manager.db.models import JobInfo, JobQueue, DB
from arch.task_manager.settings import logger, PARTY_ID, WORK_MODE
import errno
from arch.api import eggroll
import datetime
import json
import threading


class IdCounter:
    _lock = threading.RLock()

    def __init__(self, initial_value=0):
        self._value = initial_value

    def incr(self, delta=1):
        '''
        Increment the counter with locking
        '''
        with IdCounter._lock:
            self._value += delta
            return self._value


id_counter = IdCounter()


def generate_job_id():
    return '_'.join([datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), str(PARTY_ID), str(id_counter.incr())])


def get_job_directory(job_id=None):
    _paths = ['jobs', job_id] if job_id else ['jobs']
    return os.path.join(file_utils.get_project_base_directory(), *_paths)


def new_runtime_conf(job_dir, method, module, role, party_id):
    if role:
        conf_path_dir = os.path.join(job_dir, method, module, role, str(party_id))
    else:
        conf_path_dir = os.path.join(job_dir, method, module, str(party_id))
    os.makedirs(conf_path_dir, exist_ok=True)
    return os.path.join(conf_path_dir, 'runtime_conf.json')


@DB.connection_context()
def save_job_info(job_id, role, party_id, save_info, create=False):
    jobs = JobInfo.select().where(JobInfo.job_id == job_id, JobInfo.role == role, JobInfo.party_id == party_id)
    is_insert = True
    if jobs:
        job_info = jobs[0]
        is_insert = False
    elif create:
        job_info = JobInfo()
    else:
        return None
    job_info.job_id = job_id
    job_info.role = role
    job_info.party_id = party_id
    job_info.create_date = datetime.datetime.now()
    for k, v in save_info.items():
        if k in ['job_id', 'role', 'party_id']:
            continue
        setattr(job_info, k, v)
    if is_insert:
        job_info.save(force_insert=True)
    else:
        job_info.save()
    return job_info


@DB.connection_context()
def set_job_failed(job_id, role, party_id):
    sql = JobInfo.update(status='failed').where(JobInfo.job_id == job_id,
                                                JobInfo.role == role,
                                                JobInfo.party_id == party_id,
                                                JobInfo.status != 'success')
    return sql.execute() > 0


@DB.connection_context()
def query_job_by_id(job_id):
    jobs = JobInfo.select().where(JobInfo.job_id == job_id)
    return [job for job in jobs]


@DB.connection_context()
def push_into_job_queue(job_id, config):
    job_queue = JobQueue()
    job_queue.job_id = job_id
    job_queue.status = 'waiting'
    job_queue.config = json.dumps(config)
    job_queue.party_id = PARTY_ID
    job_queue.create_date = datetime.datetime.now()
    job_queue.save(force_insert=True)


@DB.connection_context()
def get_job_from_queue(status, limit=1):
    if limit:
        jobs = JobQueue.select().where(JobQueue.status == status, JobQueue.party_id == PARTY_ID).order_by(JobQueue.create_date.asc()).limit(limit)
    else:
        jobs = JobQueue.select().where(JobQueue.status == status, JobQueue.party_id == PARTY_ID).order_by(JobQueue.create_date.asc())
    return [job for job in jobs]


@DB.connection_context()
def update_job_queue(job_id, role, party_id, save_data):
    jobs = JobQueue.select().where(JobQueue.job_id == job_id, JobQueue.role == role, JobQueue.party_id == party_id)
    is_insert = True
    if jobs:
        job_queue = jobs[0]
        is_insert = False
    else:
        job_queue = JobQueue()
        job_queue.create_date = datetime.datetime.now()
    job_queue.job_id = job_id
    job_queue.role = role
    job_queue.party_id = party_id
    for k, v in save_data.items():
        if k in ['job_id', 'role', 'party_id']:
            continue
        setattr(job_queue, k, v)
    if is_insert:
        job_queue.save(force_insert=True)
    else:
        job_queue.save()
    return job_queue


@DB.connection_context()
def pop_from_job_queue(job_id):
    try:
        query = JobQueue.delete().where(JobQueue.job_id == job_id)
        return query.execute() > 0
    except Exception as e:
        return False


@DB.connection_context()
def job_queue_size():
    return JobQueue.select().count()


@DB.connection_context()
def show_job_queue():
    jobs = JobQueue.select().where(JobQueue.role == 'guest', JobQueue.pid.is_null(False)).distinct()
    return [job for job in jobs]


@DB.connection_context()
def running_job_amount():
    return JobQueue.select().where(JobQueue.status == "running", JobQueue.pid.is_null(False)).distinct().count()


def is_job_initiator(initiator, party_id=PARTY_ID):
    if not initiator or not party_id:
        return False
    return int(initiator) == int(party_id)


def clean_job(job_id):
    try:
        logger.info('ready clean job {}'.format(job_id))
        eggroll.cleanup('*', namespace=job_id, persistent=False)
        logger.info('send clean job {}'.format(job_id))
    except Exception as e:
        logger.exception(e)


def gen_status_id():
    return uuid.uuid1().hex


def check_job_process(pid):
    if pid < 0:
        return False
    if pid == 0:
        raise ValueError('invalid PID 0')
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        return True


def run_subprocess(job_dir, job_role, progs):
    logger.info('Starting progs: {}'.format(progs))
    logger.info(' '.join(progs))

    std_dir = os.path.join(job_dir, job_role)
    if not os.path.exists(std_dir):
        os.makedirs(os.path.join(job_dir, job_role))
    std_log = open(os.path.join(std_dir, 'std.log'), 'w')
    task_pid_path = os.path.join(job_dir, 'pids')

    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    else:
        startupinfo = None
    p = subprocess.Popen(progs,
                         stdout=std_log,
                         stderr=std_log,
                         startupinfo=startupinfo
                         )
    os.makedirs(task_pid_path, exist_ok=True)
    with open(os.path.join(task_pid_path, job_role + ".pid"), 'w') as f:
        f.truncate()
        f.write(str(p.pid) + "\n")
        f.flush()
    return p
