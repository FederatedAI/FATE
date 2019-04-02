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
from flask import Flask, request
import os
from arch.task_manager.utils.job_utils import get_json_result, get_job_directory
from arch.task_manager.job_manager import save_job_info, update_job_queue, update_job_info
import subprocess
from arch.api.utils import file_utils
import json
import datetime
import psutil
from psutil import NoSuchProcess

manager = Flask(__name__)


@manager.route('/<job_id>/<module>/<role>', methods=['POST'])
def start_workflow(job_id, module, role):
    _data = request.json
    _job_dir = get_job_directory(job_id)
    _party_id = str(_data['local']['party_id'])
    _method = _data['WorkFlowParam']['method']
    conf_path_dir = os.path.join(_job_dir, _method, module, role, _party_id)
    os.makedirs(conf_path_dir, exist_ok=True)
    conf_file_path = os.path.join(conf_path_dir, 'runtime_conf.json')
    with open(conf_file_path, 'w+') as f:
        f.truncate()
        f.write(json.dumps(_data, indent=4))
        f.flush()
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    else:
        startupinfo = None
    task_pid_path = os.path.join(_job_dir, 'pids')
    std_log = open(os.path.join(_job_dir, role + '.std.log'), 'w')

    progs = ["python3",
             os.path.join(file_utils.get_project_base_directory(), _data['CodePath']),
             "-j", job_id,
             "-c", os.path.abspath(conf_file_path)
             ]
    print(" ".join(progs))
    manager.logger.info('Starting progs: {}'.format(" ".join(progs)))

    p = subprocess.Popen(progs,
                         stdout=std_log,
                         stderr=std_log,
                         startupinfo=startupinfo
                         )
    os.makedirs(task_pid_path, exist_ok=True)
    with open(os.path.join(task_pid_path, role + ".pid"), 'w') as f:
        f.truncate()
        f.write(str(p.pid) + "\n")
        f.flush()

    job_data = dict()
    job_data["begin_date"] = datetime.datetime.now()
    job_data["status"] = "ready"
    with open(conf_file_path) as fr:
        config = json.load(fr)
    job_data.update(config)
    save_job_info(job_id=job_id, **job_data)
    update_job_queue(job_id=job_id, update_data={"status": "ready"})
    return get_json_result(msg="success, pid is %s" % p.pid)


@manager.route('/<job_id>', methods=['DELETE'])
def stop_workflow(job_id):
    _job_dir = get_job_directory(job_id)
    task_pid_path = os.path.join(_job_dir, 'pids')
    if os.path.isdir(task_pid_path):
        for pid_file in os.listdir(task_pid_path):
            try:
                if not pid_file.endswith('.pid'):
                    continue
                with open(os.path.join(task_pid_path, pid_file), 'r') as f:
                    pids = f.read().split('\n')
                    for pid in pids:
                        try:
                            if len(pid) == 0:
                                continue
                            manager.logger.debug("terminating process pid:{} {}".format(pid, pid_file))
                            p = psutil.Process(int(pid))
                            for child in p.children(recursive=True):
                                child.kill()
                            p.kill()
                        except NoSuchProcess:
                            continue
            except Exception as e:
                manager.logger.exception("error")
                continue
        update_job_info(job_id=job_id, update_data={"status": "failed", "set_status": "failed"})
    return get_json_result()
