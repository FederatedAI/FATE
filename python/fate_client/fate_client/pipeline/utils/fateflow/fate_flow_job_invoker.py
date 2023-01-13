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
import time
from datetime import timedelta
from .flow_client import FlowClient
from ...conf.env_config import FlowConfig


class FATEFlowJobInvoker(object):
    def __init__(self):
        self._client = FlowClient(ip=FlowConfig.IP, port=FlowConfig.PORT, version=FlowConfig.VERSION)

    def monitor_status(self, job_id, role, party_id):
        start_time = time.time()
        pre_task = None
        print(f"Job id is {job_id}\n")
        while True:
            response_data = self.query_job(job_id, role, party_id)
            status = response_data["status"]
            if status == JobStatus.SUCCESS:
                elapse_seconds = timedelta(seconds=int(time.time() - start_time))
                print(f"Job is success!!! Job id is {job_id}, response_data={response_data}")
                print(f"Total time: {elapse_seconds}")
                break

            elif status == JobStatus.RUNNING:
                code, data = self.query_task(job_id=job_id, role=role, party_id=party_id,
                                             status=JobStatus.RUNNING)

                if code != 0 or len(data) == 0:
                    time.sleep(0.1)
                    continue

                elapse_seconds = timedelta(seconds=int(time.time() - start_time))
                if len(data) == 1:
                    task = data[0]["task_name"]
                else:
                    task = []
                    for task_data in data:
                        task.append(task_data["task_name"])

                if task != pre_task:
                    print(f"\r")
                    pre_task = task
                print(f"\x1b[80D\x1b[1A\x1b[KRunning task {task}, time elapse: {elapse_seconds}")

            elif status == JobStatus.WAITING:
                elapse_seconds = timedelta(seconds=int(time.time() - start_time))
                print(f"\x1b[80D\x1b[1A\x1b[KJob is waiting, time elapse: {elapse_seconds}")

            elif status in [JobStatus.FAILED, JobStatus.CANCELED]:
                raise ValueError(f"Job is {status}, please check out job_id={job_id} in fate_flow log directory")

            time.sleep(1)

    def submit_job(self, dag_schema):
        response = self._client.submit_job(dag_schema=dag_schema)
        try:
            code = response["code"]
            if code != 0:
                raise ValueError(f"Return code {code}!=0")

            job_id = response["job_id"]
            model_id = response["data"]["model_id"]
            model_version = response["data"]["model_version"]
            return job_id, model_id, model_version
        except BaseException:
            raise ValueError(f"submit job is failed, response={response}")

    def query_job(self, job_id, role, party_id):
        response = self._client.query_job(job_id, role, party_id)
        try:
            code = response["code"]
            if code != 0:
                raise ValueError(f"Return code {code}!=0")

            data = response["data"][0]
            return data
        except BaseException:
            raise ValueError(f"query job is failed, response={response}")

    def query_task(self, job_id, role, party_id, status):
        response = self._client.query_task(job_id, role, party_id, status)
        try:
            code = response["code"]
            data = response.get("data", [])
            return code, data
        except BaseException:
            raise ValueError(f"query task is failed, response={response}")

    def query_site_info(self):
        response = self._client.query_site_info()
        try:
            code = response["code"]
            if code != 0:
                return None

            party_id = response["data"]["party_id"]
            return party_id
        except ValueError:
            return None

    def upload_data(self, upload_conf):
        response = self._client.upload_data(upload_conf)
        try:
            code = response["code"]
            if code != 0:
                raise ValueError(f"Return code {code}!=0")

            namespace = response["data"]["namespace"]
            name = response["data"]["name"]
            print(f"Upload data successfully, please use eggroll:///{namespace}/{name} as input uri")
        except BaseException:
            raise ValueError(f"Upload data fails, response={response}")

    def get_output_data(self, ):
        ...

    def get_output_model(self, job_id, role, party_id, task_name):
        response = self._client.query_model(job_id, role, party_id, task_name)
        try:
            code = response["code"]
            if code != 0:
                raise ValueError(f"Return code {code}!=0")
            model = response["data"]["output_model"]
            return model
        except BaseException:
            raise ValueError(f"query task={job_id}, role={role}, "
                             f"party_id={party_id}'s output model is failed, response={response}")

    def get_output_metrics(self, job_id, role, party_id, task_name):
        response = self._client.query_metrics(job_id, role, party_id, task_name)
        try:
            code = response["code"]
            if code != 0:
                raise ValueError(f"Return code {code}!=0")
            metrics = response["data"]
            return metrics
        except BaseException:
            raise ValueError(f"query task={job_id}, role={role}, "
                             f"party_id={party_id}'s output metrics is failed, response={response}")


class JobStatus(object):
    WAITING = 'waiting'
    READY = 'ready'
    RUNNING = "running"
    CANCELED = "canceled"
    TIMEOUT = "timeout"
    FAILED = "failed"
    PASS = "pass"
    SUCCESS = "success"
