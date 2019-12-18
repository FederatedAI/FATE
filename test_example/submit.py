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

import json
import os
import subprocess
import sys
import tempfile
import time
from string import Template


class Submitter(object):

    def __init__(self):
        self._fate_home = ""
        self._flow_client_path = ""
        self._work_mode = 0
        self._backend = 0

    def set_fate_home(self, path):
        self._fate_home = path
        self._flow_client_path = os.path.join(path, "fate_flow/fate_flow_client.py")
        return self

    def set_work_mode(self, mode):
        self._work_mode = mode
        return self

    def set_backend(self, backend):
        self._backend = backend

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
        print(str(stdout))
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
                scp_out = self.run_cmd(["scp", f.name, f"{remote_host}:{f.name}"])
                env_path = os.path.join(self._fate_home, "../init_env.sh")
                print(scp_out)
                upload_cmd = " && ".join([f"source {env_path}"
                                          f"python {self._flow_client_path} -f upload -c {f.name}",
                                          f"rm {f.name}"])
                upload_out = self.run_cmd(["ssh", remote_host, upload_cmd])
                print(upload_out)
            else:
                self.submit(["-f", "upload", "-c", f.name])

    def delete_table(self, namespace, name):
        pass

    def run_upload(self, data_path, config, remote_host=None):
        conf = dict(
            file=data_path,
            head=config["head"],
            partition=config["partition"],
            work_mode=self._work_mode,
            table_name=config["table_name"],
            namespace=config["namespace"]
        )
        with tempfile.NamedTemporaryFile("w") as f:
            json.dump(conf, f)
            f.flush()
            if remote_host:
                scp_out = self.run_cmd(["scp", f.name, f"{remote_host}:{f.name}"])
                env_path = os.path.join(self._fate_home, "../init_env.sh")
                # print(scp_out)
                print(env_path)
                print(f.name)
                print(self._flow_client_path)
                upload_cmd = " && ".join([f"source {env_path}",
                                          f"python {self._flow_client_path} -f upload -c {f.name}",
                                          f"rm {f.name}"])
                upload_out = self.run_cmd(["ssh", remote_host, upload_cmd])
                print(upload_out)
            else:
                self.submit(["-f", "upload", "-c", f.name])

    def submit_job(self, conf_temperate_path, dsl_path, **substitutes):
        conf = self.render(conf_temperate_path, **substitutes)
        with tempfile.NamedTemporaryFile("w") as f:
            json.dump(conf, f)
            f.flush()
            stdout = self.submit(["-f", "submit_job", "-c", f.name, "-d", dsl_path])
        result = {}
        result['jobId'] = stdout["jobId"]
        result['model_info'] = stdout["data"]["model_info"]
        return result

    def submit_pre_job(self, conf_temperate_path, model_info, **substitutes):
        conf = self.model_render(conf_temperate_path, model_info, **substitutes)
        with tempfile.NamedTemporaryFile("w") as f:
            json.dump(conf, f)
            f.flush()
            stdout = self.submit(["-f", "submit_job", "-c", f.name])
        return stdout["jobId"]

    def fix_config(self, file_path, **substitutes):
        pass

    def render(self, conf_temperate_path, **substitutes):
        temp = open(conf_temperate_path).read()
        substituted = Template(temp).substitute(**substitutes)
        d = json.loads(substituted)
        d['job_parameters']['work_mode'] = self._work_mode
        return d

    def model_render(self, conf_temperate_path, model_info, **substitutes):
        temp = open(conf_temperate_path).read()
        substituted = Template(temp).substitute(**substitutes)
        d = json.loads(substituted)
        d['job_parameters']['work_mode'] = self._work_mode
        d['job_parameters']['model_id'] = model_info['model_id']
        d['job_parameters']['model_version'] = model_info['model_version']
        return d

    def await_finish(self, job_id, timeout=sys.maxsize, check_interval=10):
        deadline = time.time() + timeout
        while True:
            time.sleep(check_interval)
            stdout = self.submit(["-f", "query_job", "-j", job_id, "-r", "guest"])
            status = stdout["data"][0]["f_status"]
            if status == "running" and time.time() < deadline:
                continue
            else:
                return status
