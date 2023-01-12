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
import requests


class Address(object):
    SUBMIT_JOB = "/job/submit"
    QUERY_JOB = "/job/query"
    STOP_JOB = "/job/stop"
    QUERY_TASK = "/job/task/query"
    SITE_INFO = "/site/info/query"
    UPLOAD_DATA = "/data/upload"
    QUERY_MODEL = "/output/model/query"
    QUERY_METRIC = "/output/metric/query"

    def __init__(self, server_url):
        self._submit_job_url = server_url + self.SUBMIT_JOB
        self._query_job_url = server_url + self.QUERY_JOB
        self._stop_job_url = server_url + self.STOP_JOB
        self._query_task_url = server_url + self.QUERY_TASK

        self._site_info_url = server_url + self.SITE_INFO

        self._upload_url = server_url + self.UPLOAD_DATA

        self._query_model_url = server_url + self.QUERY_MODEL
        self._query_metric_url = server_url + self.QUERY_METRIC

    @property
    def submit_job_url(self):
        return self._submit_job_url

    @property
    def query_job_url(self):
        return self._query_job_url

    @property
    def stop_job_url(self):
        return self._stop_job_url

    @property
    def query_task_url(self):
        return self._query_task_url

    @property
    def site_info_url(self):
        return self._site_info_url

    @property
    def upload_data_url(self):
        return self._upload_url

    @property
    def query_model_url(self):
        return self._query_model_url

    @property
    def query_metric_url(self):
        return self._query_metric_url


class FlowClient(object):
    def __init__(self, ip, port, version):
        self._server_url = f"http://{ip}:{port}/{version}"
        self._address = Address(server_url=self._server_url)

    def submit_job(self, dag_schema):
        response = requests.post(
            self._address.submit_job_url,
            json={
                "dag_schema": dag_schema
            }
        )

        return response.json()

    def query_job(self, job_id, role, party_id):
        response = requests.post(
            self._address.query_job_url,
            json={
                "job_id": job_id,
                "role": role,
                "party_id": party_id
            }
        )

        return response.json()

    def stop_job(self, job_id):
        response = requests.post(
            self._address.stop_job_url,
            json={
                "job_id": job_id
            }
        )

        return response.json()

    def query_task(self, job_id, role, party_id, status):
        response = requests.post(
            self._address.query_task_url,
            json={
                "job_id": job_id,
                "role": role,
                "party_id": party_id,
                "status": status
            }
        )

        return response.json()

    def query_site_info(self):
        response = requests.get(
            self._address.site_info_url
        )
        return response.json()

    def upload_data(self, upload_conf):
        response = requests.post(
            self._address.upload_data_url,
            json=upload_conf
        )

        return response.json()

    def query_model(self, job_id, role, party_id, task_name):
        resource = requests.get(
            self._address.query_model_url,
            params={
                "job_id": job_id,
                "role": role,
                "party_id": party_id,
                "task_name": task_name
            }
        )

        return resource.json()

    def query_metrics(self, job_id, role, party_id, task_name):
        resource = requests.get(
            self._address.query_metric_url,
            params={
                "job_id": job_id,
                "role": role,
                "party_id": party_id,
                "task_name": task_name
            }
        )

        return resource.json()
