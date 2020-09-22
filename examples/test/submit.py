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
import logging


def set_logger(name):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(filename=f"{name}.cmd.log", level=logging.DEBUG, format=log_format, datefmt=date_format)


class Submitter(object):

    def __init__(self, fate_home, work_mode, backend, existing_strategy, spark_submit_config):
        self._fate_home = fate_home
        self._work_mode = work_mode
        self._backend = backend
        self._existing_strategy = existing_strategy
        self._spark_submit_config = spark_submit_config

    @property
    def _flow_client_path(self):
        return os.path.join(self._fate_home, "../python/fate_flow/fate_flow_client.py")

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
        logging.info(f"cmd: {' '.join(cmd)}")
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
            if status == 100 and "table already exists" in stdout["retmsg"]:
                return None
            raise ValueError(f"[submit_job]fail, status:{status}, stdout:{stdout}")
        return stdout

    def upload(self, data_path, namespace, name, partition=10, head=1, remote_host=None, backend=None):
        if backend is None:
            backend = self._backend
        conf = dict(
            file=data_path,
            head=head,
            partition=partition,
            work_mode=self._work_mode,
            table_name=name,
            backend=backend,
            namespace=namespace
        )
        with tempfile.NamedTemporaryFile("w") as f:
            json.dump(conf, f)
            f.flush()
            if remote_host:
                self.run_cmd(["scp", f.name, f"{remote_host}:{f.name}"])
                env_path = os.path.join(self._fate_home, "../bin/init_env.sh")
                upload_cmd = f"source {env_path}"
                upload_cmd = f"{upload_cmd} && python {self._flow_client_path} -f upload -c {f.name}"
                if self._existing_strategy == 0 or self._existing_strategy == 1:
                    upload_cmd = f"{upload_cmd} -drop {self._existing_strategy}"
                upload_cmd = f"{upload_cmd} && rm {f.name}"
                stdout = self.run_cmd(["ssh", remote_host, upload_cmd])
                try:
                    stdout = json.loads(stdout)
                    status = stdout["retcode"]
                except json.decoder.JSONDecodeError:
                    raise ValueError(f"[submit_job]fail, stdout:{stdout}")
                if status != 0:
                    if status == 100 and "table already exists" in stdout["retmsg"]:
                        return None
                    raise ValueError(f"[submit_job]fail, status:{status}, stdout:{stdout}")
                return stdout["jobId"]
            else:
                cmd = ["-f", "upload", "-c", f.name]
                if self._existing_strategy == 0 or self._existing_strategy == 1:
                    cmd.extend(["-drop", f"{self._existing_strategy}"])
                stdout = self.submit(cmd)
                if stdout is None:
                    return None
                else:
                    return stdout["jobId"]

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
        d['job_parameters']['backend'] = self._backend
        d['job_parameters']['spark_submit_config'] = self._spark_submit_config
        initiator_role = d['initiator']['role']
        d['initiator']['party_id'] = roles[initiator_role][0]
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
