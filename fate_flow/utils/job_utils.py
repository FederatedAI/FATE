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
import functools
import errno
import json
import operator
import os
import subprocess
import sys
import threading
import typing
import uuid

import psutil
from fate_flow.entity.constant import JobStatus

from fate_arch.common import file_utils
from fate_arch.common.base_utils import json_loads, json_dumps, fate_uuid, current_timestamp
from fate_arch.common.log import schedule_logger
from fate_flow.scheduler.dsl_parser import DSLParser, DSLParserV2
from fate_flow.db.db_models import DB, Job, Task
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.settings import stat_logger, JOB_DEFAULT_TIMEOUT, WORK_MODE
from fate_flow.utils import detect_utils
from fate_flow.utils import api_utils
from fate_flow.utils import session_utils
from flask import request, redirect, url_for
from fate_flow.operation.job_saver import JobSaver
from fate_flow.entity.constant import TaskStatus


class IdCounter(object):
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


def generate_federated_id(task_id, task_version):
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


def check_pipeline_job_runtime_conf(runtime_conf: typing.Dict):
    detect_utils.check_config(runtime_conf, ['initiator', 'job_parameters', 'role'])
    detect_utils.check_config(runtime_conf['initiator'], ['role', 'party_id'])
    detect_utils.check_config(runtime_conf['job_parameters'], [('work_mode', RuntimeConfig.WORK_MODE)])
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


def save_job_conf(job_id, job_dsl, job_runtime_conf, train_runtime_conf, pipeline_dsl):
    path_dict = get_job_conf_path(job_id=job_id)
    os.makedirs(os.path.dirname(path_dict.get('job_dsl_path')), exist_ok=True)
    for data, conf_path in [(job_dsl, path_dict['job_dsl_path']), (job_runtime_conf, path_dict['job_runtime_conf_path']),
                            (train_runtime_conf, path_dict['train_runtime_conf_path']), (pipeline_dsl, path_dict['pipeline_dsl_path'])]:
        with open(conf_path, 'w+') as f:
            f.truncate()
            if not data:
                data = {}
            f.write(json_dumps(data, indent=4))
            f.flush()
    return path_dict


def get_job_conf_path(job_id):
    job_dir = get_job_directory(job_id)
    job_dsl_path = os.path.join(job_dir, 'job_dsl.json')
    job_runtime_conf_path = os.path.join(job_dir, 'job_runtime_conf.json')
    train_runtime_conf_path = os.path.join(job_dir, 'train_runtime_conf.json')
    pipeline_dsl_path = os.path.join(job_dir, 'pipeline_dsl.json')
    return {'job_dsl_path': job_dsl_path,
            'job_runtime_conf_path': job_runtime_conf_path,
            'train_runtime_conf_path': train_runtime_conf_path,
            'pipeline_dsl_path': pipeline_dsl_path}


def get_job_conf(job_id):
    conf_dict = {}
    for key, path in get_job_conf_path(job_id).items():
        config = file_utils.load_json_conf(path)
        conf_dict[key] = config
    return conf_dict


def get_job_dsl_parser_by_job_id(job_id):
    with DB.connection_context():
        jobs = Job.select(Job.f_dsl, Job.f_runtime_conf, Job.f_train_runtime_conf).where(Job.f_job_id == job_id)
        if jobs:
            job = jobs[0]
            job_dsl_parser = get_job_dsl_parser(dsl=job.f_dsl, runtime_conf=job.f_runtime_conf,
                                                train_runtime_conf=job.f_train_runtime_conf)
            return job_dsl_parser
        else:
            return None


def get_job_dsl_parser(dsl=None, runtime_conf=None, pipeline_dsl=None, train_runtime_conf=None):
    # dsl_parser = DSLParser()
    parser_version = str(runtime_conf.get('job_parameters', {}).get('dsl_version', '1'))
    dsl_parser = get_dsl_parser_by_version(parser_version)
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


def get_parser_version_mapping():
    return {
        "1": DSLParser(),
        "2": DSLParserV2()
    }


def get_dsl_parser_by_version(version: str = "1"):
    mapping = get_parser_version_mapping()
    if version not in mapping:
        raise Exception("{} version of dsl parser is not currently supported.".format(version))
    return mapping[version]


def get_job_configuration(job_id, role, party_id, tasks=None):
    with DB.connection_context():
        if tasks:
            jobs_run_conf = {}
            for task in tasks:
                jobs = Job.select(Job.f_job_id, Job.f_runtime_conf, Job.f_description).where(Job.f_job_id == task.f_job_id)
                job = jobs[0]
                jobs_run_conf[job.f_job_id] = job.f_runtime_conf["role_parameters"]["local"]["upload_0"]
                jobs_run_conf[job.f_job_id]["notes"] = job.f_description
            return jobs_run_conf
        else:
            jobs = Job.select(Job.f_dsl, Job.f_runtime_conf, Job.f_train_runtime_conf).where(Job.f_job_id == job_id,
                                                                                             Job.f_role == role,
                                                                                             Job.f_party_id == party_id)
        if jobs:
            job = jobs[0]
            return job.f_dsl, job.f_runtime_conf, job.f_train_runtime_conf
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


def list_job(limit):
    with DB.connection_context():
        jobs = Job.select().order_by(Job.f_create_time.desc()).limit(limit)
        return [job for job in jobs]


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


def list_task(limit):
    with DB.connection_context():
        tasks = Task.select().order_by(Task.f_create_time.desc()).limit(limit)
        return [task for task in tasks]


def success_task_count(job_id):
    count = 0
    tasks = query_task(job_id=job_id)
    job_component_status = {}
    for task in tasks:
        job_component_status[task.f_component_name] = job_component_status.get(task.f_component_name, set())
        job_component_status[task.f_component_name].add(task.f_status)
    for component_name, role_status in job_component_status.items():
        if len(role_status) == 1 and JobStatus.COMPLETE in role_status:
            count += 1
    return count


def update_job_progress(job_id, dag, current_task_id):
    role, party_id = query_job_info(job_id)
    component_count = len(dag.get_dependency(role=role, party_id=int(party_id))['component_list'])
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
            schedule_logger(task.f_job_id).info("job {} task {} {} {} no process pid".format(
                task.f_job_id, task.f_task_id, task.f_role, task.f_party_id))
            return True
        pid = int(task.f_run_pid)
        schedule_logger(task.f_job_id).info("try to stop job {} task {} {} {} process pid:{}".format(
            task.f_job_id, task.f_task_id, task.f_role, task.f_party_id, pid))
        if not check_job_process(pid):
            schedule_logger(task.f_job_id).info("can not found job {} task {} {} {} process pid:{}".format(
                task.f_job_id, task.f_task_id, task.f_role, task.f_party_id, pid))
            return True
        p = psutil.Process(int(pid))
        if not is_task_executor_process(task=task, process=p):
            schedule_logger(task.f_job_id).warning("this pid {} is not job {} task {} {} {} executor".format(
                pid, task.f_job_id, task.f_task_id, task.f_role, task.f_party_id))
            return False
        for child in p.children(recursive=True):
            if check_job_process(child.pid) and is_task_executor_process(task=task, process=child):
                child.stop_job()
        if not only_child:
            if check_job_process(p.pid) and is_task_executor_process(task=task, process=p):
                p.kill()
        schedule_logger(task.f_job_id).info("successfully stop job {} task {} {} {} process pid:{}".format(
            task.f_job_id, task.f_task_id, task.f_role, task.f_party_id, pid))
        return True
    except Exception as e:
        raise e


def start_clean_queue():
    schedule_logger().info('get clean queue command')
    jobs = JobSaver.query_job(is_initiator=1, status=JobStatus.WAITING)
    if jobs:
        for job in jobs:
            schedule_logger(job.f_job_id).info(
                'start send {} job {} command success'.format(JobStatus.CANCELED, job.f_job_id))
            job_info = {'f_job_id': job.f_job_id, 'f_status': JobStatus.CANCELED}
            JobSaver.update_job(job_info=job_info)
            job_runtime_conf = json_loads(job.f_runtime_conf)
            event = job_event(job.f_job_id,
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


def start_session_stop(task):
    job_conf_dict = get_job_conf(task.f_job_id)
    runtime_conf = job_conf_dict['job_runtime_conf_path']
    computing_session_id = generate_session_id(task.f_task_id, task.f_task_version, task.f_role, task.f_party_id, suffix="computing")
    storage_session_id = generate_session_id(task.f_task_id, task.f_task_version, task.f_role, task.f_party_id, suffix="storage")
    if task.f_status != TaskStatus.WAITING:
        schedule_logger(task.f_job_id).info('start run subprocess to stop task {} {} session {} and {}'
                                            .format(task.f_task_id, task.f_task_version, computing_session_id, storage_session_id))
    else:
        schedule_logger(task.f_job_id).info('task {} {} is waiting, pass stop session {} and {}'
                                            .format(task.f_task_id, task.f_task_version, computing_session_id, storage_session_id))
        return
    task_dir = os.path.join(get_job_directory(job_id=task.f_job_id), task.f_role,
                            task.f_party_id, task.f_component_name, 'session_stop')
    os.makedirs(task_dir, exist_ok=True)
    process_cmd = [
        'python3', sys.modules[session_utils.SessionStop.__module__].__file__,
        '-j', computing_session_id,
        '-w', str(runtime_conf.get('job_parameters').get('work_mode')),
        '-b', str(runtime_conf.get('job_parameters').get('backend', 0)),
        '-c', 'stop' if task.f_status == JobStatus.COMPLETE else 'kill'
    ]
    p = run_subprocess(config_dir=task_dir, process_cmd=process_cmd, log_dir=None)
    process_cmd[3] = storage_session_id
    p = run_subprocess(config_dir=task_dir, process_cmd=process_cmd, log_dir=None)


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


# TODO: support task executor routing
def job_server_routing(routing_type=0):
    def _out_wrapper(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            job_server = set()
            jobs = query_job(job_id=request.json.get('job_id', None))
            for job in jobs:
                if job.f_run_ip:
                    job_server.add(job.f_run_ip)
            if len(job_server) == 1:
                execute_host = job_server.pop()
                if execute_host != RuntimeConfig.JOB_SERVER_HOST:
                    if routing_type == 0:
                        return api_utils.request_execute_server(request=request, execute_host=execute_host)
                    else:
                        return redirect('http://{}{}'.format(execute_host, url_for(request.endpoint)), code=307)
            return func(*args, **kwargs)
        return _wrapper
    return _out_wrapper


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


def job_event(job_id, initiator_role,  initiator_party_id):
    event = {'job_id': job_id,
             "initiator_role": initiator_role,
             "initiator_party_id": initiator_party_id
             }
    return event


def get_task_info(job_id, role, party_id, component_name):
    task_info = {
        'job_id': job_id,
        'role': role,
        'party_id': party_id
    }
    if component_name:
        task_info['component_name'] = component_name
    return task_info


def query_job_info(job_id):
    jobs = query_job(job_id=job_id, is_initiator=1)
    party_id = None
    role = None
    if jobs:
        job = jobs[0]
        role = job.f_role
        party_id = job.f_party_id
    return role, party_id


def cleaning(signum, frame):
    sys.exit(0)


def federation_cleanup(job, task):
    from fate_flow.entity.constant import Backend, StoreEngine
    from fate_arch.common import Party

    runtime_conf = json_loads(job.f_runtime_conf)
    job_parameters = runtime_conf['job_parameters']
    backend = Backend(job_parameters.get('backend', 0))
    store_engine = StoreEngine(job_parameters.get('store_engine', 0))

    if backend.is_spark() and store_engine.is_hdfs():
        runtime_conf['local'] = {'role': job.f_role, 'party_id': job.f_party_id}
        parties = [Party(k, p) for k,v in runtime_conf['role'].items() for p in v ]
        from fate_arch.session.spark import Session
        ssn = Session(session_id=task.f_task_id)
        ssn.init_federation(federation_session_id=task.f_task_id, runtime_conf=runtime_conf)
        ssn._get_federation().generate_mq_names(parties=parties)
        ssn._get_federation().cleanup()


