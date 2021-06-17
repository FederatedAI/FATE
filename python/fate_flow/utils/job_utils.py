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
import os
import subprocess
import sys
import threading
import typing

import psutil

from fate_arch.common import file_utils
from fate_arch.common.base_utils import json_dumps, fate_uuid, current_timestamp
from fate_arch.common.log import schedule_logger
from fate_flow.db.db_models import DB, Job, Task
from fate_flow.entity.types import JobStatus
from fate_flow.entity.types import TaskStatus, RunParameters, KillProcessStatusCode
from fate_flow.settings import stat_logger, JOB_DEFAULT_TIMEOUT, WORK_MODE, FATE_BOARD_DASHBOARD_ENDPOINT
from fate_flow.utils import detect_utils, model_utils
from fate_flow.utils import session_utils
from fate_arch.common.conf_utils import get_base_config


class JobIdGenerator(object):
    _lock = threading.RLock()

    def __init__(self, initial_value=0):
        self._value = initial_value
        self._pre_timestamp = None
        self._max = 99999

    def next_id(self):
        '''
        generate next job id with locking
        '''
        #todo: there is duplication in the case of multiple instances deployment
        now = datetime.datetime.now()
        with JobIdGenerator._lock:
            if self._pre_timestamp == now:
                if self._value < self._max:
                    self._value += 1
                else:
                    now += datetime.timedelta(microseconds=1)
                    self._pre_timestamp = now
                    self._value = 0
            else:
                self._pre_timestamp = now
                self._value = 0
            return "{}{}".format(now.strftime("%Y%m%d%H%M%S%f"), self._value)


job_id_generator = JobIdGenerator()


def generate_job_id():
    return job_id_generator.next_id()


def generate_task_id(job_id, component_name):
    return '{}_{}'.format(job_id, component_name)


def generate_task_version_id(task_id, task_version):
    return "{}_{}".format(task_id, task_version)


def generate_session_id(task_id, task_version, role, party_id, suffix=None, random_end=False):
    items = [task_id, str(task_version), role, str(party_id)]
    if suffix:
        items.append(suffix)
    if random_end:
        items.append(fate_uuid())
    return "_".join(items)


def generate_task_input_data_namespace(task_id, task_version, role, party_id):
    return "input_data_{}".format(generate_session_id(task_id=task_id,
                                                      task_version=task_version,
                                                      role=role,
                                                      party_id=party_id))


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


def check_job_runtime_conf(runtime_conf: typing.Dict):
    detect_utils.check_config(runtime_conf, ['initiator', 'job_parameters', 'role'])
    detect_utils.check_config(runtime_conf['initiator'], ['role', 'party_id'])
    # deal party id
    runtime_conf['initiator']['party_id'] = int(runtime_conf['initiator']['party_id'])
    for r in runtime_conf['role'].keys():
        for i in range(len(runtime_conf['role'][r])):
            runtime_conf['role'][r][i] = int(runtime_conf['role'][r][i])


def runtime_conf_basic(if_local=False):
    job_runtime_conf = {
        "initiator": {},
        "job_parameters": {"work_mode": WORK_MODE},
        "role": {},
        "role_parameters": {}
    }
    if if_local:
        job_runtime_conf["initiator"]["role"] = "local"
        job_runtime_conf["initiator"]["party_id"] = 0
        job_runtime_conf["role"]["local"] = [0]
    return job_runtime_conf


def new_runtime_conf(job_dir, method, module, role, party_id):
    if role:
        conf_path_dir = os.path.join(job_dir, method, module, role, str(party_id))
    else:
        conf_path_dir = os.path.join(job_dir, method, module, str(party_id))
    os.makedirs(conf_path_dir, exist_ok=True)
    return os.path.join(conf_path_dir, 'runtime_conf.json')


def save_job_conf(job_id, role, job_dsl, job_runtime_conf, job_runtime_conf_on_party, train_runtime_conf, pipeline_dsl=None):
    path_dict = get_job_conf_path(job_id=job_id, role=role)
    os.makedirs(os.path.dirname(path_dict.get('job_dsl_path')), exist_ok=True)
    os.makedirs(os.path.dirname(path_dict.get('job_runtime_conf_on_party_path')), exist_ok=True)
    for data, conf_path in [(job_dsl, path_dict['job_dsl_path']),
                            (job_runtime_conf, path_dict['job_runtime_conf_path']),
                            (job_runtime_conf_on_party, path_dict['job_runtime_conf_on_party_path']),
                            (train_runtime_conf, path_dict['train_runtime_conf_path']),
                            (pipeline_dsl, path_dict['pipeline_dsl_path'])]:
        with open(conf_path, 'w+') as f:
            f.truncate()
            if not data:
                data = {}
            f.write(json_dumps(data, indent=4))
            f.flush()
    return path_dict


def get_job_conf_path(job_id, role):
    job_dir = get_job_directory(job_id)
    job_dsl_path = os.path.join(job_dir, 'job_dsl.json')
    job_runtime_conf_path = os.path.join(job_dir, 'job_runtime_conf.json')
    job_runtime_conf_on_party_path = os.path.join(job_dir, role, 'job_runtime_on_party_conf.json')
    train_runtime_conf_path = os.path.join(job_dir, 'train_runtime_conf.json')
    pipeline_dsl_path = os.path.join(job_dir, 'pipeline_dsl.json')
    return {'job_dsl_path': job_dsl_path,
            'job_runtime_conf_path': job_runtime_conf_path,
            'job_runtime_conf_on_party_path': job_runtime_conf_on_party_path,
            'train_runtime_conf_path': train_runtime_conf_path,
            'pipeline_dsl_path': pipeline_dsl_path}


def get_job_conf(job_id, role):
    conf_dict = {}
    for key, path in get_job_conf_path(job_id, role).items():
        config = file_utils.load_json_conf(path)
        conf_dict[key] = config
    return conf_dict


@DB.connection_context()
def get_job_configuration(job_id, role, party_id, tasks=None):
    if tasks:
        jobs_run_conf = {}
        for task in tasks:
            jobs = Job.select(Job.f_job_id, Job.f_runtime_conf_on_party, Job.f_description).where(Job.f_job_id == task.f_job_id)
            job = jobs[0]
            jobs_run_conf[job.f_job_id] = job.f_runtime_conf_on_party["component_parameters"]["role"]["local"]["0"]["upload_0"]
            jobs_run_conf[job.f_job_id]["notes"] = job.f_description
        return jobs_run_conf
    else:
        jobs = Job.select(Job.f_dsl, Job.f_runtime_conf, Job.f_train_runtime_conf, Job.f_runtime_conf_on_party).where(Job.f_job_id == job_id,
                                                                                                                      Job.f_role == role,
                                                                                                                      Job.f_party_id == party_id)
    if jobs:
        job = jobs[0]
        return job.f_dsl, job.f_runtime_conf, job.f_runtime_conf_on_party, job.f_train_runtime_conf
    else:
        return {}, {}, {}, {}


@DB.connection_context()
def get_model_configuration(job_id, role, party_id):
    res = model_utils.query_model_info(model_version=job_id, role=role, party_id=party_id,
                                       query_filters=['train_dsl', 'dsl', 'train_runtime_conf', 'runtime_conf'])
    if res:
        dsl = res[0].get('train_dsl') if res[0].get('train_dsl') else res[0].get('dsl')
        runtime_conf = res[0].get('runtime_conf')
        train_runtime_conf = res[0].get('train_runtime_conf')
        return dsl, runtime_conf, train_runtime_conf
    return {}, {}, {}


    # models = MLModel.select(MLModel.f_dsl, MLModel.f_runtime_conf,
    #                         MLModel.f_train_runtime_conf).where(MLModel.f_job_id == job_id,
    #                                                             MLModel.f_role == role,
    #                                                             MLModel.f_party_id == party_id)
    # if models:
    #     model = models[0]
    #     return model.f_dsl, model.f_runtime_conf, model.f_train_runtime_conf
    # else:
    #     return {}, {}, {}


@DB.connection_context()
def get_job_parameters(job_id, role, party_id):
    jobs = Job.select(Job.f_runtime_conf_on_party).where(Job.f_job_id == job_id,
                                                         Job.f_role == role,
                                                         Job.f_party_id == party_id)
    if jobs:
        job = jobs[0]
        return job.f_runtime_conf_on_party.get("job_parameters")
    else:
        return {}


@DB.connection_context()
def get_job_dsl(job_id, role, party_id):
    jobs = Job.select(Job.f_dsl).where(Job.f_job_id == job_id,
                                       Job.f_role == role,
                                       Job.f_party_id == party_id)
    if jobs:
        job = jobs[0]
        return job.f_dsl
    else:
        return {}


def job_virtual_component_name():
    return "pipeline"


def job_virtual_component_module_name():
    return "Pipeline"


@DB.connection_context()
def list_job(limit):
    if limit > 0:
        jobs = Job.select().order_by(Job.f_create_time.desc()).limit(limit)
    else:
        jobs = Job.select().order_by(Job.f_create_time.desc())
    return [job for job in jobs]


@DB.connection_context()
def list_task(limit):
    if limit > 0:
        tasks = Task.select().order_by(Task.f_create_time.desc()).limit(limit)
    else:
        tasks = Task.select().order_by(Task.f_create_time.desc())
    return [task for task in tasks]


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


def check_job_is_timeout(job: Job):
    job_parameters = job.f_runtime_conf_on_party["job_parameters"]
    timeout = job_parameters.get("timeout", JOB_DEFAULT_TIMEOUT)
    now_time = current_timestamp()
    running_time = (now_time - job.f_create_time)/1000
    if running_time > timeout:
        schedule_logger(job_id=job.f_job_id).info('job {}  run time {}s timeout'.format(job.f_job_id, running_time))
        return True
    else:
        return False


def check_process_by_keyword(keywords):
    if not keywords:
        return True
    keyword_filter_cmd = ' |'.join(['grep %s' % keyword for keyword in keywords])
    ret = os.system('ps aux | {} | grep -v grep | grep -v "ps aux "'.format(keyword_filter_cmd))
    return ret == 0


def run_subprocess(job_id, config_dir, process_cmd, log_dir=None, job_dir=None):
    schedule_logger(job_id=job_id).info('start process command: {}'.format(' '.join(process_cmd)))

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
                         startupinfo=startupinfo,
                         cwd=job_dir,
                         )
    with open(pid_path, 'w') as f:
        f.truncate()
        f.write(str(p.pid) + "\n")
        f.flush()
    schedule_logger(job_id=job_id).info('start process command: {} successfully, pid is {}'.format(' '.join(process_cmd), p.pid))
    return p


def wait_child_process(signum, frame):
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


def is_task_executor_process(task: Task, process: psutil.Process):
    """
    check the process if task executor or not by command
    :param task:
    :param process:
    :return:
    """
    # Todo: The same map should be used for run task command
    run_cmd_map = {
        3: "f_job_id",
        5: "f_component_name",
        7: "f_task_id",
        9: "f_task_version",
        11: "f_role",
        13: "f_party_id"
    }
    try:
        cmdline = process.cmdline()
        schedule_logger(task.f_job_id).info(cmdline)
    except Exception as e:
        # Not sure whether the process is a task executor process, operations processing is required
        schedule_logger(task.f_job_id).warning(e)
        return False
    for i, k in run_cmd_map.items():
        if len(cmdline) > i and cmdline[i] == str(getattr(task, k)):
            continue
        else:
            # todo: The logging level should be obtained first
            if len(cmdline) > i:
                schedule_logger(task.f_job_id).debug(f"cmd map {i} {k}, cmd value {cmdline[i]} task value {getattr(task, k)}")
            return False
    else:
        return True


def kill_task_executor_process(task: Task, only_child=False):
    try:
        if not task.f_run_pid:
            schedule_logger(task.f_job_id).info("job {} task {} {} {} with {} party status no process pid".format(
                task.f_job_id, task.f_task_id, task.f_role, task.f_party_id, task.f_party_status))
            return KillProcessStatusCode.NOT_FOUND
        pid = int(task.f_run_pid)
        schedule_logger(task.f_job_id).info("try to stop job {} task {} {} {} with {} party status process pid:{}".format(
            task.f_job_id, task.f_task_id, task.f_role, task.f_party_id, task.f_party_status, pid))
        if not check_job_process(pid):
            schedule_logger(task.f_job_id).info("can not found job {} task {} {} {} with {} party status process pid:{}".format(
                task.f_job_id, task.f_task_id, task.f_role, task.f_party_id, task.f_party_status, pid))
            return KillProcessStatusCode.NOT_FOUND
        p = psutil.Process(int(pid))
        if not is_task_executor_process(task=task, process=p):
            schedule_logger(task.f_job_id).warning("this pid {} is not job {} task {} {} {} executor".format(
                pid, task.f_job_id, task.f_task_id, task.f_role, task.f_party_id))
            return KillProcessStatusCode.ERROR_PID
        for child in p.children(recursive=True):
            if check_job_process(child.pid) and is_task_executor_process(task=task, process=child):
                child.kill()
        if not only_child:
            if check_job_process(p.pid) and is_task_executor_process(task=task, process=p):
                p.kill()
        schedule_logger(task.f_job_id).info("successfully stop job {} task {} {} {} process pid:{}".format(
            task.f_job_id, task.f_task_id, task.f_role, task.f_party_id, pid))
        return KillProcessStatusCode.KILLED
    except Exception as e:
        raise e


def start_session_stop(task):
    job_parameters = RunParameters(**get_job_parameters(job_id=task.f_job_id, role=task.f_role, party_id=task.f_party_id))
    computing_session_id = generate_session_id(task.f_task_id, task.f_task_version, task.f_role, task.f_party_id)
    if task.f_status != TaskStatus.WAITING:
        schedule_logger(task.f_job_id).info(f'start run subprocess to stop task session {computing_session_id}')
    else:
        schedule_logger(task.f_job_id).info(f'task is waiting, pass stop session {computing_session_id}')
        return
    task_dir = os.path.join(get_job_directory(job_id=task.f_job_id), task.f_role,
                            task.f_party_id, task.f_component_name, 'session_stop')
    os.makedirs(task_dir, exist_ok=True)
    process_cmd = [
        'python3', sys.modules[session_utils.SessionStop.__module__].__file__,
        '-j', computing_session_id,
        '--computing', job_parameters.computing_engine,
        '--federation', job_parameters.federation_engine,
        '--storage', job_parameters.storage_engine,
        '-c', 'stop' if task.f_status == JobStatus.SUCCESS else 'kill'
    ]
    p = run_subprocess(job_id=task.f_job_id, config_dir=task_dir, process_cmd=process_cmd, log_dir=None)


def get_timeout(job_id, timeout, runtime_conf, dsl):
    try:
        if timeout > 0:
            schedule_logger(job_id).info('setting job {} timeout {}'.format(job_id, timeout))
            return timeout
        else:
            default_timeout = job_default_timeout(runtime_conf, dsl)
            schedule_logger(job_id).info('setting job {} timeout {} not a positive number, using the default timeout {}'.format(
                job_id, timeout, default_timeout))
            return default_timeout
    except:
        default_timeout = job_default_timeout(runtime_conf, dsl)
        schedule_logger(job_id).info('setting job {} timeout {} is incorrect, using the default timeout {}'.format(
            job_id, timeout, default_timeout))
        return default_timeout


def job_default_timeout(runtime_conf, dsl):
    # future versions will improve
    timeout = JOB_DEFAULT_TIMEOUT
    return timeout


def get_board_url(job_id, role, party_id):
    board_url = "http://{}:{}{}".format(
        get_base_config("fateboard", {}).get("host"),
        get_base_config("fateboard", {}).get("port"),
        FATE_BOARD_DASHBOARD_ENDPOINT).format(job_id, role, party_id)
    return board_url

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=50) as t:
        obj_list = []
        for page in range(1, 50):
            obj = t.submit(new_generate_job_id)
            obj_list.append(obj)

        for future in as_completed(obj_list):
            data = future.result()
            print(f"main: {data}")