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
import datetime
import errno
import json
import operator
import os
import subprocess
import threading
import typing
import uuid

import psutil

from arch.api.utils import file_utils
from arch.api.utils.core import current_timestamp
from arch.api.utils.core import json_loads, json_dumps
from fate_flow.db.db_models import DB, Job, Task
from fate_flow.driver.dsl_parser import DSLParser
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.settings import stat_logger, WORK_MODE
from fate_flow.utils import detect_utils


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
    return '{}{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"), str(id_counter.incr()))


def generate_task_id(job_id, component_name):
    return '{}_{}'.format(job_id, component_name)


def get_job_directory(job_id):
    return os.path.join(file_utils.get_project_base_directory(), 'jobs', job_id)


def get_job_log_directory(job_id):
    return os.path.join(file_utils.get_project_base_directory(), 'logs', job_id)


def check_config(config: typing.Dict, required_parameters: typing.List):
    for parameter in required_parameters:
        if parameter not in config:
            return False, 'configuration no {} parameter'.format(parameter)
    else:
        return True, 'ok'


def check_pipeline_job_runtime_conf(runtime_conf: typing.Dict):
    detect_utils.check_config(runtime_conf, ['initiator', 'job_parameters', 'role'])
    detect_utils.check_config(runtime_conf['initiator'], ['role', 'party_id'])
    detect_utils.check_config(runtime_conf['job_parameters'], [('work_mode', WORK_MODE)])
    # deal party id
    runtime_conf['initiator']['party_id'] = int(runtime_conf['initiator']['party_id'])
    for r in runtime_conf['role'].keys():
        for i in range(len(runtime_conf['role'][r])):
            runtime_conf['role'][r][i] = int(runtime_conf['role'][r][i])


def new_runtime_conf(job_dir, method, module, role, party_id):
    if role:
        conf_path_dir = os.path.join(job_dir, method, module, role, str(party_id))
    else:
        conf_path_dir = os.path.join(job_dir, method, module, str(party_id))
    os.makedirs(conf_path_dir, exist_ok=True)
    return os.path.join(conf_path_dir, 'runtime_conf.json')


def save_job_conf(job_id, job_dsl, job_runtime_conf):
    job_dsl_path, job_runtime_conf_path = get_job_conf_path(job_id=job_id)
    os.makedirs(os.path.dirname(job_dsl_path), exist_ok=True)
    for data, conf_path in [(job_dsl, job_dsl_path), (job_runtime_conf, job_runtime_conf_path)]:
        with open(conf_path, 'w+') as f:
            f.truncate()
            f.write(json.dumps(data, indent=4))
            f.flush()
    return job_dsl_path, job_runtime_conf_path


def get_job_conf_path(job_id):
    job_dir = get_job_directory(job_id)
    job_dsl_path = os.path.join(job_dir, 'job_dsl.json')
    job_runtime_conf_path = os.path.join(job_dir, 'job_runtime_conf.json')
    return job_dsl_path, job_runtime_conf_path


def get_job_dsl_parser_by_job_id(job_id):
    with DB.connection_context():
        jobs = Job.select(Job.f_dsl, Job.f_runtime_conf, Job.f_train_runtime_conf).where(Job.f_job_id == job_id)
        if jobs:
            job = jobs[0]
            job_dsl_parser = get_job_dsl_parser(dsl=json_loads(job.f_dsl), runtime_conf=json_loads(job.f_runtime_conf),
                                                train_runtime_conf=json_loads(job.f_train_runtime_conf))
            return job_dsl_parser
        else:
            return None


def get_job_dsl_parser(dsl=None, runtime_conf=None, pipeline_dsl=None, train_runtime_conf=None):
    dsl_parser = DSLParser()
    default_runtime_conf_path = os.path.join(file_utils.get_project_base_directory(),
                                             *['federatedml', 'conf', 'default_runtime_conf'])
    setting_conf_path = os.path.join(file_utils.get_project_base_directory(), *['federatedml', 'conf', 'setting_conf'])
    job_type = runtime_conf.get('job_parameters', {}).get('job_type', 'train')
    dsl_parser.run(dsl=dsl,
                   runtime_conf=runtime_conf,
                   pipeline_dsl=pipeline_dsl,
                   pipeline_runtime_conf=train_runtime_conf,
                   default_runtime_conf_prefix=default_runtime_conf_path,
                   setting_conf_prefix=setting_conf_path,
                   mode=job_type)
    return dsl_parser


def get_job_configuration(job_id, role, party_id):
    with DB.connection_context():
        jobs = Job.select(Job.f_dsl, Job.f_runtime_conf, Job.f_train_runtime_conf).where(Job.f_job_id == job_id,
                                                                                         Job.f_role == role,
                                                                                         Job.f_party_id == party_id)
        if jobs:
            job = jobs[0]
            return json_loads(job.f_dsl), json_loads(job.f_runtime_conf), json_loads(job.f_train_runtime_conf)
        else:
            return {}, {}, {}


def query_job(**kwargs):
    with DB.connection_context():
        filters = []
        for f_n, f_v in kwargs.items():
            attr_name = 'f_%s' % f_n
            if hasattr(Job, attr_name):
                filters.append(operator.attrgetter('f_%s' % f_n)(Job) == f_v)
        if filters:
            jobs = Job.select().where(*filters)
            return [job for job in jobs]
        else:
            # not allow query all job
            return []


def job_queue_size():
    return RuntimeConfig.JOB_QUEUE.qsize()


def show_job_queue():
    # TODO
    pass


def query_task(**kwargs):
    with DB.connection_context():
        filters = []
        for f_n, f_v in kwargs.items():
            attr_name = 'f_%s' % f_n
            if hasattr(Task, attr_name):
                filters.append(operator.attrgetter('f_%s' % f_n)(Task) == f_v)
        if filters:
            tasks = Task.select().where(*filters)
        else:
            tasks = Task.select()
        return [task for task in tasks]


def success_task_count(job_id):
    count = 0
    tasks = query_task(job_id=job_id)
    job_component_status = {}
    for task in tasks:
        job_component_status[task.f_component_name] = job_component_status.get(task.f_component_name, set())
        job_component_status[task.f_component_name].add(task.f_status)
    for component_name, role_status in job_component_status.items():
        if len(role_status) == 1 and 'success' in role_status:
            count += 1
    return count


def update_job_progress(job_id, dag, current_task_id):
    component_count = len(dag.get_dependency()['component_list'])
    success_count = success_task_count(job_id=job_id)
    job = Job()
    job.f_progress = float(success_count) / component_count * 100
    job.f_update_time = current_timestamp()
    job.f_current_tasks = json_dumps([current_task_id])
    return job


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


def check_process_by_keyword(keywords):
    if not keywords:
        return True
    keyword_filter_cmd = ' |'.join(['grep %s' % keyword for keyword in keywords])
    ret = os.system('ps aux | {} | grep -v grep | grep -v "ps aux "'.format(keyword_filter_cmd))
    return ret == 0


def run_subprocess(config_dir, process_cmd, log_dir=None):
    stat_logger.info('Starting process command: {}'.format(process_cmd))
    stat_logger.info(' '.join(process_cmd))

    os.makedirs(config_dir, exist_ok=True)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    std_log = open(os.path.join(log_dir if log_dir else config_dir, 'std.log'), 'w')
    pid_path = os.path.join(config_dir, 'pid')

    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    else:
        startupinfo = None
    p = subprocess.Popen(process_cmd,
                         stdout=std_log,
                         stderr=std_log,
                         startupinfo=startupinfo
                         )
    with open(pid_path, 'w') as f:
        f.truncate()
        f.write(str(p.pid) + "\n")
        f.flush()
    return p


def wait_child_process(signum, frame):
    child_pid = None
    try:
        while True:
            child_pid, status = os.waitpid(-1, os.WNOHANG)
            if child_pid == 0:
                stat_logger.info('no child process was immediately available')
                break
            exitcode = status >> 8
            stat_logger.info('child process %s exit with exitcode %s', child_pid, exitcode)
    except OSError as e:
        if e.errno == errno.ECHILD:
            stat_logger.warning('current process has no existing unwaited-for child processes.')
        else:
            raise


def kill_process(pid):
    try:
        if not pid:
            return False
        stat_logger.info("terminating process pid:{}".format(pid))
        if not check_job_process(pid):
            return True
        p = psutil.Process(int(pid))
        for child in p.children(recursive=True):
            if check_job_process(child.pid):
                child.kill()
        if check_job_process(p.pid):
            p.kill()
        return True
    except Exception as e:
        raise e


def gen_all_party_key(all_party):
    """
    Join all party as party key
    :param all_party:
        "role": {
            "guest": [9999],
            "host": [10000],
            "arbiter": [10000]
         }
    :return:
    """
    if not all_party:
        all_party_key = 'all'
    elif isinstance(all_party, dict):
        sorted_role_name = sorted(all_party.keys())
        all_party_key = '#'.join([
            ('%s-%s' % (
                role_name,
                '_'.join([str(p) for p in sorted(set(all_party[role_name]))]))
             )
            for role_name in sorted_role_name])
    else:
        all_party_key = None
    return all_party_key
