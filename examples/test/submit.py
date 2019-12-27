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

import collections.abc
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta


class Submitter(object):

    def __init__(self, fate_home="", work_mode=0, backend=0):
        self._fate_home = fate_home
        self._work_mode = work_mode
        self._backend = backend

    @property
    def _flow_client_path(self):
        return os.path.join(self._fate_home, "../fate_flow/fate_flow_client.py")

    def set_fate_home(self, path):
        self._fate_home = path
        return self

    def set_work_mode(self, mode):
        self._work_mode = mode
        return self

    def set_backend(self, backend):
        self._backend = backend
        return self

    @staticmethod
    def run_cmd(cmd):
        subp = subprocess.Popen(cmd,
                                shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        stdout, stderr = subp.communicate()
        return stdout.decode("utf-8")

    def submit(self, cmd):
        full_cmd = ["python", self._flow_client_path]
        full_cmd.extend(cmd)
        stdout = self.run_cmd(full_cmd)
        try:
            stdout = json.loads(stdout)
            status = stdout["retcode"]
        except json.decoder.JSONDecodeError:
            raise ValueError(f"[submit_job]fail, stdout:{stdout}")
        if status != 0:
            raise ValueError(f"[submit_job]fail, status:{status}, stdout:{stdout}")
        return stdout

    def upload(self, data_path, namespace, name, partition=10, head=1, remote_host=None):
        conf = dict(
            file=data_path,
            head=head,
            partition=partition,
            work_mode=self._work_mode,
            table_name=name,
            namespace=namespace
        )
        with tempfile.NamedTemporaryFile("w") as f:
            json.dump(conf, f)
            f.flush()
            if remote_host:
                self.run_cmd(["scp", f.name, f"{remote_host}:{f.name}"])
                env_path = os.path.join(self._fate_home, "../../init_env.sh")
                upload_cmd = " && ".join([f"source {env_path}",
                                          f"python {self._flow_client_path} -f upload -c {f.name}",
                                          f"rm {f.name}"])
                stdout = self.run_cmd(["ssh", remote_host, upload_cmd])
                try:
                    stdout = json.loads(stdout)
                    status = stdout["retcode"]
                except json.decoder.JSONDecodeError:
                    raise ValueError(f"[submit_job]fail, stdout:{stdout}")
                if status != 0:
                    raise ValueError(f"[submit_job]fail, status:{status}, stdout:{stdout}")
                return stdout
            else:
                return self.submit(["-f", "upload", "-c", f.name])

    def delete_table(self, namespace, name):
        pass

    def submit_job(self, conf_path, roles, submit_type="train", dsl_path=None, model_info=None, substitute=None):
        conf = self.render(conf_path, roles, model_info, substitute)
        result = {}
        with tempfile.NamedTemporaryFile("w") as f:
            json.dump(conf, f)
            f.flush()
            if submit_type == "train":
                stdout = self.submit(["-f", "submit_job", "-c", f.name, "-d", dsl_path])
                result['model_info'] = stdout["data"]["model_info"]
            else:
                stdout = self.submit(["-f", "submit_job", "-c", f.name])
            result['jobId'] = stdout["jobId"]
        return result

    def render(self, conf_path, roles, model_info=None, substitute=None):
        with open(conf_path) as f:
            d = json.load(f)
        if substitute is not None:
            d = recursive_update(d, substitute)
        d['job_parameters']['work_mode'] = self._work_mode
        d['initiator']['party_id'] = roles["guest"][0]
        for r in ["guest", "host", "arbiter"]:
            if r in d['role']:
                for idx in range(len(d['role'][r])):
                    d['role'][r][idx] = roles[r][idx]
        if model_info is not None:
            d['job_parameters']['model_id'] = model_info['model_id']
            d['job_parameters']['model_version'] = model_info['model_version']
        return d

    def await_finish(self, job_id, timeout=sys.maxsize, check_interval=3, task_name=None):
        deadline = time.time() + timeout
        start = time.time()
        while True:
            stdout = self.submit(["-f", "query_job", "-j", job_id])
            status = stdout["data"][0]["f_status"]
            elapse_seconds = int(time.time() - start)
            date = time.strftime('%Y-%m-%d %X')
            if task_name:
                log_msg = f"[{date}][{task_name}]{status}, elapse: {timedelta(seconds=elapse_seconds)}"
            else:
                log_msg = f"[{date}]{job_id} {status}, elapse: {timedelta(seconds=elapse_seconds)}"
            if (status == "running" or status == "waiting") and time.time() < deadline:
                print(log_msg, end="\r")
                time.sleep(check_interval)
                continue
            else:
                print(" " * 60, end="\r")  # clean line
                print(log_msg)
                return status


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
