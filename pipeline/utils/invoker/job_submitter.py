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
import tempfile
import time

from fate_flow.flowpy.client import FlowClient
from pipeline.backend import config as conf
from pipeline.backend.config import JobStatus
from pipeline.backend.config import StatusCode


# FATE_HOME = os.getcwd() + "/../../"
# FATE_FLOW_CLIENT = FATE_HOME + "fate_flow/fate_flow_client.py"


class JobFunc:
    SUBMIT_JOB = "submit_job"
    UPLOAD = "upload"
    COMPONENT_OUTPUT_MODEL = "component_output_model"
    COMPONENT_METRIC = "component_metric_all"
    JOB_STATUS = "query_job"
    TASK_STATUS = "query_task"
    COMPONENT_OUTPUT_DATA = "component_output_data"
    COMPONENT_OUTPUT_DATA_TABLE = "component_output_data_table"
    DEPLOY_COMPONENT = "deo"


class JobInvoker(object):
    def __init__(self):
        self.client = FlowClient()

    @classmethod
    def _run_cmd(cls, cmd, output_while_running=False):
        subp = subprocess.Popen(cmd,
                                shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        if not output_while_running:
            stdout, stderr = subp.communicate()
            return stdout.decode("utf-8")
        else:
            for line in subp.stdout:
                if line == "":
                    continue
                else:
                    print(line.strip())

    def submit_job(self, dsl=None, submit_conf=None):
        dsl_path = None
        with tempfile.TemporaryDirectory() as job_dir:
            if dsl:
                dsl_path = os.path.join(job_dir, "job_dsl.json")
                import pprint
                pprint.pprint(dsl)
                with open(dsl_path, "w") as fout:
                    fout.write(json.dumps(dsl))

            submit_path = os.path.join(job_dir, "job_runtime_conf.json")
            with open(submit_path, "w") as fout:
                fout.write(json.dumps(submit_conf))

            result = self.client.job.submit(conf_path=submit_path, dsl_path=dsl_path)
            try:
                # result = json.loads(result)
                if 'retcode' not in result or result["retcode"] != 0:
                    raise ValueError

                if "jobId" not in result:
                    raise ValueError

                job_id = result["jobId"]
                data = result["data"]
            except ValueError:
                raise ValueError("job submit failed, err msg: {}".format(result))

        return job_id, data

    def upload_data(self, submit_conf=None, drop=0):
        with tempfile.TemporaryDirectory() as job_dir:
            submit_path = os.path.join(job_dir, "job_runtime_conf.json")
            with open(submit_path, "w") as fout:
                fout.write(json.dumps(submit_conf))

            result = self.client.data.upload(conf_path=submit_path, verbose=1, drop=drop)
            try:
                # result = json.loads(result)
                if 'retcode' not in result or result["retcode"] != 0:
                    raise ValueError

                if "jobId" not in result:
                    raise ValueError

                job_id = result["jobId"]
                data = result["data"]
            except ValueError:
                raise ValueError("job submit failed, err msg: {}".format(result))

        return job_id, data

    def monitor_job_status(self, job_id, role, party_id):
        party_id = str(party_id)
        while True:
            ret_code, ret_msg, data = self.query_job(job_id, role, party_id)
            status = data["f_status"]
            if status == JobStatus.COMPLETE:
                print("job is success!!!")
                return StatusCode.SUCCESS

            if status == JobStatus.FAILED:
                print("job is failed, please check out job {} by fate board or fate_flow cli".format(job_id))
                return StatusCode.FAIL

            if status == JobStatus.WAITING:
                print("job {} is still waiting")

            if status == JobStatus.RUNNING:
                print("job {} now is running component {}".format(job_id, data["f_current_tasks"]))

            time.sleep(conf.TIME_QUERY_FREQS)

    def query_job(self, job_id, role, party_id):
        party_id=str(party_id)
        result = self.client.job.query(job_id=job_id, role=role, party_id=party_id)
        try:
            # result = json.loads(result)
            if 'retcode' not in result:
                raise ValueError("can not query_job")

            ret_code = result["retcode"]
            ret_msg = result["retmsg"]
            print(f"query job result is {result}")
            data = result["data"][0]
            return ret_code, ret_msg, data
        except ValueError:
            raise ValueError("query job result is {}, can not parse useful info".format(result))

    def get_output_data_table(self, job_id, cpn_name, role, party_id):
        party_id=str(party_id)
        result = self.client.component.output_data_table(job_id=job_id, role=role,
                                                         party_id=party_id, component_name=cpn_name)
        try:
            # result = json.loads(result)
            if 'retcode' not in result or result["retcode"] != 0:
                raise ValueError

            if "data" not in result:
                raise ValueError
            data = result["data"]
        except ValueError:
            raise ValueError("job submit failed, err msg: {}".format(result))
        return data

    def query_task(self, job_id, cpn_name, role, party_id):
        party_id=str(party_id)
        result = self.client.task.query(job_id=job_id, role=role,
                                        party_id=party_id, component_name=cpn_name)
        try:
            # result = json.loads(result)
            if 'retcode' not in result:
                raise ValueError("can not query component {}' task status".format(cpn_name))

            ret_code = result["retcode"]
            ret_msg = result["retmsg"]

            data = result["data"]
            return ret_code, ret_msg, data
        except ValueError:
            raise ValueError("query task result is {}, can not parse useful info".format(result))

    def get_output_data(self, job_id, cpn_name, role, party_id, limits=None):
        party_id = str(party_id)
        with tempfile.TemporaryDirectory() as job_dir:
            result = self.client.component.output_data(job_id=job_id, role=role, output_path=job_dir,
                                                       party_id=party_id, component_name=cpn_name)
            # result = json.loads(result)
            output_dir = result["directory"]
            # output_data_meta = os.path.join(output_dir, "output_data_meta.json")
            n = 0
            for file in os.listdir(output_dir):
                if file.endswith("csv"):
                    n += 1
            # single output data
            if n == 1:
                output_data = os.path.join(output_dir, f"output_0_data.csv")
                data = JobInvoker.extract_output_data(output_data)
            # multiple output data
            else:
                data = []
                for i in range(n):
                    output_data = os.path.join(output_dir, f"output_{i}_data.csv")
                    data_i = JobInvoker.extract_output_data(output_data)
                    data.append(data_i)
            return data

    @staticmethod
    def extract_output_data(output_data):
        data = []
        with open(output_data, "r") as fin:
            for line in fin:
                data.append(line.strip())

        print(f"{output_data}: {data[:10]}")
        return data

    def get_model_param(self, job_id, cpn_name, role, party_id):
        result = None
        party_id = str(party_id)
        try:
            result = self.client.component.output_model(job_id=job_id, role=role,
                                                        party_id=party_id, component_name=cpn_name)
            # result = json.loads(result)
            if "data" not in result:
                print("job {}, component {} has no output model param".format(job_id, cpn_name))
                return
            return result["data"]
        except:
            print("Can not get output model, err msg is {}".format(result))

    def get_metric(self, job_id, cpn_name, role, party_id):
        result = None
        party_id = str(party_id)
        try:
            result = self.client.component.metric_all(job_id=job_id, role=role,
                                                      party_id=party_id, component_name=cpn_name)
            # result = json.loads(result)
            if "data" not in result:
                print("job {}, component {} has no output metric".format(job_id, cpn_name))
                return
            return result["data"]
        except:
            print("Can not get output model, err msg is {}".format(result))

    def get_summary(self, job_id, cpn_name, role, party_id):
        result = None
        party_id = str(party_id)
        try:
            result = self.client.component.get_summary(job_id=job_id, role=role,
                                                       party_id=party_id, component_name=cpn_name)
            if "data" not in result:
                print("job {}, component {} has no output metric".format(job_id, cpn_name))
                return
            return result["data"]
        except:
            print("Can not get output model, err msg is {}".format(result))
