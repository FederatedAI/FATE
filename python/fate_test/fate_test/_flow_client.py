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
import time
import typing
from datetime import timedelta
from pathlib import Path

import requests
from requests_toolbelt import MultipartEncoderMonitor, MultipartEncoder

from fate_test._parser import Data, Job


class FLOWClient(object):

    def __init__(self,
                 address: typing.Optional[str],
                 data_base_dir: typing.Optional[Path]):
        self.address = address
        self.version = "v1"
        self._http = requests.Session()
        self._data_base_dir = data_base_dir

    def set_address(self, address):
        self.address = address

    @property
    def _base(self):
        return f"http://{self.address}/{self.version}/"

    def upload_data(self, data: Data, callback=None) -> 'UploadDataResponse':
        try:
            response = self._upload_data(conf=data.config, verbose=0, drop=1)
            if callback is not None:
                callback(response)
            status = self._awaiting(response.job_id, "local")
            response.status = status
        except Exception as e:
            raise RuntimeError(f"upload data failed") from e
        return response

    def delete_data(self, data: Data):
        try:
            self._delete_data(table_name=data.config['table_name'], namespace=data.config['namespace'])
        except Exception as e:
            raise RuntimeError(f"delete data failed") from e

    def submit_job(self, job: Job, callback=None) -> 'SubmitJobResponse':
        try:
            response = self._submit_job(**job.submit_params)
            if callback is not None:
                callback(response)
            status = self._awaiting(response.job_id, "guest", callback)
            response.status = status

        except Exception as e:
            raise RuntimeError(f"submit job failed") from e
        return response

    def add_notes(self, job_id, role, party_id, notes):
        self._add_notes(job_id=job_id, role=role, party_id=party_id, notes=notes)

    def check_connection(self):
        try:
            version = self._http.request(method="POST", url=f"{self._base}version/get", json={"module": "FATE"},
                                         timeout=2).json()
        except Exception:
            import traceback
            traceback.print_exc()
            raise
        fate_version = version.get("data", {}).get("FATE")
        if fate_version:
            return fate_version, self.address

        raise EnvironmentError(f"connection not ok")

    def _awaiting(self, job_id, role, callback=None):
        while True:
            response = self._query_job(job_id, role=role)
            if response.status.is_done():
                return response.status
            if callback is not None:
                callback(response)
            time.sleep(1)

    def _upload_data(self, conf, verbose, drop):
        conf['drop'] = drop
        conf['verbose'] = verbose
        path = Path(conf.get('file'))
        if not path.is_file():
            path = self._data_base_dir.joinpath(conf.get('file')).resolve()

        if not path.exists():
            raise Exception('The file is obtained from the fate flow client machine, but it does not exist, '
                            f'please check the path: {path}')

        with path.open("rb") as fp:
            data = MultipartEncoder(
                fields={'file': (path.name, fp, 'application/octet-stream')}
            )
            data = MultipartEncoderMonitor(data)
            upload_response = self._post(url='data/upload', data=data, params=conf,
                                         headers={'Content-Type': data.content_type})
            response = UploadDataResponse(upload_response)
        return response

    def _delete_data(self, table_name, namespace):
        response = self._post(url='table/delete', json={'table_name': table_name, 'namespace': namespace})
        return response

    def _submit_job(self, conf, dsl):
        post_data = {
            'job_dsl': dsl,
            'job_runtime_conf': conf
        }
        response = SubmitJobResponse(self._post(url='job/submit', json=post_data))
        return response

    def _query_job(self, job_id, role):
        data = {"local": {"role": role}, "job_id": str(job_id)}
        response = QueryJobResponse(self._post(url='job/query', json=data))
        return response

    def _add_notes(self, job_id, role, party_id, notes):
        data = dict(job_id=job_id, role=role, party_id=party_id, notes=notes)
        response = AddNotesResponse(self._post(url='job/update', json=data))
        return response

    def _post(self, url, **kwargs) -> dict:
        request_url = self._base + url
        try:
            response = self._http.request(method='post', url=request_url, **kwargs)
        except Exception as e:
            raise RuntimeError(f"post {url} with {kwargs} failed") from e

        try:
            if isinstance(response, requests.models.Response):
                response = response.json()
            else:
                try:
                    response = json.loads(response.content.decode('utf-8', 'ignore'), strict=False)
                except (TypeError, ValueError):
                    return response
        except json.decoder.JSONDecodeError:
            response = {'retcode': 100,
                        'retmsg': "Internal server error. Nothing in response. You may check out the configuration in "
                                  "'FATE/arch/conf/server_conf.json' and restart fate flow server."}
        return response


class Status(object):
    def __init__(self, status: str):
        self.status = status

    def is_done(self):
        return self.status.lower() in ['complete', 'success', 'canceled', 'failed']

    def is_success(self):
        return self.status.lower() in ['complete', 'success']

    def __str__(self):
        return self.status

    def __repr__(self):
        return self.__str__()


class QueryJobResponse(object):
    def __init__(self, response: dict):
        try:
            status = Status(response.get('data')[0]["f_status"])
            progress = response.get('data')[0]['f_progress']
        except Exception as e:
            raise RuntimeError(f"query job error, response: {response}") from e
        self.status = status
        self.progress = progress


class UploadDataResponse(object):
    def __init__(self, response: dict):
        try:
            self.job_id = response["jobId"]
        except Exception as e:
            raise RuntimeError(f"upload error, response: {response}") from e
        self.status: typing.Optional[Status] = None


class AddNotesResponse(object):
    def __init__(self, response: dict):
        try:
            retcode = response['retcode']
            retmsg = response['retmsg']
            if retcode != 0 or retmsg != 'success':
                raise RuntimeError(f"add notes error: {response}")
        except Exception as e:
            raise RuntimeError(f"add notes error: {response}") from e


class SubmitJobResponse(object):
    def __init__(self, response: dict):
        try:
            self.job_id = response["jobId"]
            self.model_info = response["data"]["model_info"]
        except Exception as e:
            raise RuntimeError(f"submit job error, response: {response}") from e
        self.status: typing.Optional[Status] = None


class DataProgress(object):
    def __init__(self, role_str):
        self.role_str = role_str
        self.start = time.time()
        self.show_str = f"[{self.elapse()}] {self.role_str}"
        self.job_id = ""

    def elapse(self):
        return f"{timedelta(seconds=int(time.time() - self.start))}"

    def submitted(self, job_id):
        self.job_id = job_id
        self.show_str = f"[{self.elapse()}]{self.job_id} {self.role_str}"

    def update(self):
        self.show_str = f"[{self.elapse()}]{self.job_id} {self.role_str}"

    def show(self):
        return self.show_str


class JobProgress(object):
    def __init__(self, name):
        self.name = name
        self.start = time.time()
        self.show_str = f"[{self.elapse()}] {self.name}"
        self.job_id = ""

    def elapse(self):
        return f"{timedelta(seconds=int(time.time() - self.start))}"

    def submitted(self, job_id):
        self.job_id = job_id
        self.show_str = f"[{self.elapse()}]{self.job_id} submitted {self.name}"

    def running(self, status, progress):
        if progress is None:
            progress = 0
        self.show_str = f"[{self.elapse()}]{self.job_id} {status} {progress:3}% {self.name}"

    def exception(self, exception_id):
        self.show_str = f"[{self.elapse()}]{self.name} exception({exception_id}): {self.job_id}"

    def final(self, status):
        self.show_str = f"[{self.elapse()}]{self.job_id} {status} {self.name}"

    def show(self):
        return self.show_str
