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
import time
import typing
from datetime import timedelta
from pathlib import Path

from fate_client.flow_sdk import FlowClient
from fate_test._parser import Data

from fate_test import _config


class FLOWClient(object):

    def __init__(self,
                 address: typing.Optional[str],
                 data_base_dir: typing.Optional[Path],
                 cache_directory: typing.Optional[Path]):
        self.address = address
        self.version = "v2"
        self._client = FlowClient(self.address.split(':')[0], self.address.split(':')[1], self.version)
        self._data_base_dir = data_base_dir
        self._cache_directory = cache_directory
        self.data_size = 0

    def set_address(self, address):
        self.address = address

    def transform_local_file_to_dataframe(self, data: Data, callback=None, output_path=None):
        data_warehouse = self.upload_data(data, callback, output_path)
        status = self.transform_to_dataframe(data.namespace, data.table_name, data_warehouse, callback)
        return status

    def upload_data(self, data: Data, callback=None, output_path=None):
        response, file_path = self._upload_data(data, output_path=output_path)
        try:
            if callback is not None:
                callback(response)
            code = response["code"]
            if code != 0:
                raise ValueError(f"Return code {code}!=0")

            namespace = response["data"]["namespace"]
            name = response["data"]["name"]
            job_id = response["job_id"]
        except BaseException:
            raise ValueError(f"Upload data fails, response={response}")
        # self.monitor_status(job_id, role=self.role, party_id=self.party_id)
        self._awaiting(job_id, "local", 0)

        return dict(namespace=namespace, name=name)

    def transform_to_dataframe(self, namespace, table_name, data_warehouse, callback=None):
        response = self._client.data.dataframe_transformer(namespace=namespace,
                                                           name=table_name,
                                                           data_warehouse=data_warehouse)

        """try:
            code = response["code"]
            if code != 0:
                raise ValueError(f"Return code {code}!=0")
            job_id = response["job_id"]
        except BaseException:
            raise ValueError(f"Transform data fails, response={response}")"""
        try:
            if callback is not None:
                callback(response)
                status = self._awaiting(response["job_id"], "local", 0)
                status = str(status).lower()
            else:
                status = response["retmsg"]

        except Exception as e:
            raise RuntimeError(f"upload data failed") from e
        job_id = response["job_id"]
        self._awaiting(job_id, "local", 0)
        return status

    def delete_data(self, data: Data):
        try:
            table_name = data.config['table_name'] if data.config.get(
                'table_name', None) is not None else data.config.get('name')
            self._client.table.delete(table_name=table_name, namespace=data.config['namespace'])
        except Exception as e:
            raise RuntimeError(f"delete data failed") from e

    def table_query(self, table_name, namespace):
        result = self._table_query(table_name=table_name, namespace=namespace)
        return result

    def add_notes(self, job_id, role, party_id, notes):
        self._add_notes(job_id=job_id, role=role, party_id=party_id, notes=notes)

    """def check_connection(self):
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

        raise EnvironmentError(f"connection not ok")"""

    def _awaiting(self, job_id, role, party_id, callback=None):
        while True:
            response = self._query_job(job_id, role=role, party_id=party_id)
            if response.status.is_done():
                return response.status
            if callback is not None:
                callback(response)
            time.sleep(1)

    def _upload_data(self, data, output_path=None, verbose=0, destroy=1):
        conf = data.config
        # if conf.get("engine", {}) != "PATH":
        if output_path is not None:
            conf['file'] = os.path.join(os.path.abspath(output_path), os.path.basename(conf.get('file')))
        else:
            if _config.data_switch is not None:
                conf['file'] = os.path.join(str(self._cache_directory), os.path.basename(conf.get('file')))
            else:
                conf['file'] = os.path.join(str(self._data_base_dir), conf.get('file'))
        path = Path(conf.get('file'))
        if not path.exists():
            raise Exception('The file is obtained from the fate flow client machine, but it does not exist, '
                            f'please check the path: {path}')
        response = self._client.data.upload(file=str(path),
                                            head=data.head,
                                            meta=data.meta,
                                            extend_sid=data.extend_sid,
                                            partitions=data.partitions)
        return response, conf["file"]

    """def _table_info(self, table_name, namespace):
        param = {
            'table_name': table_name,
            'namespace': namespace
        }
        response = self.flow_client(request='table/info', param=param)
        return response

    def _delete_data(self, table_name, namespace):
        param = {
            'table_name': table_name,
            'namespace': namespace
        }
        response = self.flow_client(request='table/delete', param=param)
        return response"""

    def _table_query(self, table_name, namespace):
        response = self._client.table.query(namespace=namespace, table_name=table_name)
        return response

    def _delete_data(self, table_name, namespace):
        response = self._client.table.delete(namespace=namespace, table_name=table_name)
        return response

    def query_job(self, job_id, role, party_id):
        response = self._client.task.query(job_id, role=role, party_id=party_id)
        return response

    """def _submit_job(self, conf, dsl):
        param = {
            'job_dsl': self._save_json(dsl, 'submit_dsl.json'),
            'job_runtime_conf': self._save_json(conf, 'submit_conf.json')
        }
        response = SubmitJobResponse(self.flow_client(request='job/submit', param=param))
        return response"""

    """def _deploy_model(self, model_id, model_version, dsl=None):
        post_data = {'model_id': model_id,
                     'model_version': model_version,
                     'predict_dsl': dsl}
        response = self.flow_client(request='model/deploy', param=post_data)
        result = {}
        try:
            retcode = response['retcode']
            retmsg = response['retmsg']
            if retcode != 0 or retmsg != 'success':
                raise RuntimeError(f"deploy model error: {response}")
            result["model_id"] = response["data"]["model_id"]
            result["model_version"] = response["data"]["model_version"]
        except Exception as e:
            raise RuntimeError(f"deploy model error: {response}") from e

        return result"""

    """def _output_data_table(self, job_id, role, party_id, component_name):
        post_data = {'job_id': job_id,
                     'role': role,
                     'party_id': party_id,
                     'component_name': component_name}
        response = self.flow_client(request='component/output_data_table', param=post_data)
        result = {}
        try:
            retcode = response['retcode']
            retmsg = response['retmsg']
            if retcode != 0 or retmsg != 'success':
                raise RuntimeError(f"deploy model error: {response}")
            result["name"] = response["data"][0]["table_name"]
            result["namespace"] = response["data"][0]["table_namespace"]
        except Exception as e:
            raise RuntimeError(f"output data table error: {response}") from e
        return result

    def _get_summary(self, job_id, role, party_id, component_name):
        post_data = {'job_id': job_id,
                     'role': role,
                     'party_id': party_id,
                     'component_name': component_name}
        response = self.flow_client(request='component/get_summary', param=post_data)
        try:
            retcode = response['retcode']
            retmsg = response['retmsg']
            result = {}
            if retcode != 0 or retmsg != 'success':
                raise RuntimeError(f"deploy model error: {response}")
            result["summary_dir"] = retmsg  # 获取summary文件位置
        except Exception as e:
            raise RuntimeError(f"output data table error: {response}") from e
        return result"""

    """def _query_job(self, job_id, role):
        param = {
            'job_id': job_id,
            'role': role
        }
        response = QueryJobResponse(self.flow_client(request='job/query', param=param))
        return response"""

    def _query_job(self, job_id, role, party_id):
        response = self._client.job.query(job_id, role, party_id)
        """try:
            code = response["code"]
            if code != 0:
                raise ValueError(f"Return code {code}!=0")

            data = response["data"][0]
            return data
        except BaseException:
            raise ValueError(f"query job is failed, response={response}")"""
        return QueryJobResponse(response)

    """def get_version(self):
        response = self._post(url='version/get', json={"module": "FATE"})
        try:
            retcode = response['retcode']
            retmsg = response['retmsg']
            if retcode != 0 or retmsg != 'success':
                raise RuntimeError(f"get version error: {response}")
            fate_version = response["data"]["FATE"]
        except Exception as e:
            raise RuntimeError(f"get version error: {response}") from e
        return fate_version"""

    def get_version(self):
        response = self._client.provider.query(name="fate")
        try:
            retcode = response['code']
            retmsg = response['message']
            if retcode != 0 or retmsg != 'success':
                raise RuntimeError(f"get version error: {response}")
            fate_version = response["data"][0]["provider_name"]
        except Exception as e:
            raise RuntimeError(f"get version error: {response}") from e
        return fate_version

    def _add_notes(self, job_id, role, party_id, notes):
        data = dict(job_id=job_id, role=role, party_id=party_id, notes=notes)
        response = AddNotesResponse(self._post(url='job/update', json=data))
        return response

    def _table_bind(self, data):
        response = self._post(url='table/bind', json=data)
        try:
            retcode = response['retcode']
            retmsg = response['retmsg']
            if retcode != 0 or retmsg != 'success':
                raise RuntimeError(f"table bind error: {response}")
        except Exception as e:
            raise RuntimeError(f"table bind error: {response}") from e
        return response


class Status(object):
    def __init__(self, status: str):
        self.status = status

    def is_done(self):
        return self.status.lower() in ['complete', 'success', 'canceled', 'failed', "timeout"]

    def is_success(self):
        return self.status.lower() in ['complete', 'success']

    def __str__(self):
        return self.status

    def __repr__(self):
        return self.__str__()


class QueryJobResponse(object):
    def __init__(self, response: dict):
        try:
            status = Status(response.get('data')[0]["status"])
            progress = response.get('data')[0]['progress']
        except Exception as e:
            raise RuntimeError(f"query job error, response: {json.dumps(response, indent=4)}") from e
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


"""class SubmitJobResponse(object):
    def __init__(self, response: dict):
        try:
            self.job_id = response["jobId"]
            self.model_info = response["data"]["model_info"]
        except Exception as e:
            raise RuntimeError(f"submit job error, response: {response}") from e
        self.status: typing.Optional[Status] = None
"""


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
        self.progress_tracking = ""

    def elapse(self):
        return f"{timedelta(seconds=int(time.time() - self.start))}"

    def set_progress_tracking(self, progress_tracking):
        self.progress_tracking = progress_tracking + " "

    def submitted(self, job_id):
        self.job_id = job_id
        self.show_str = f"{self.progress_tracking}[{self.elapse()}]{self.job_id} submitted {self.name}"

    def running(self, status, progress):
        if progress is None:
            progress = 0
        self.show_str = f"{self.progress_tracking}[{self.elapse()}]{self.job_id} {status} {progress:3}% {self.name}"

    def exception(self, exception_id):
        self.show_str = f"{self.progress_tracking}[{self.elapse()}]{self.name} exception({exception_id}): {self.job_id}"

    def final(self, status):
        self.show_str = f"{self.progress_tracking}[{self.elapse()}]{self.job_id} {status} {self.name}"

    def show(self):
        return self.show_str


class JobStatus(object):
    WAITING = 'waiting'
    READY = 'ready'
    RUNNING = "running"
    CANCELED = "canceled"
    TIMEOUT = "timeout"
    FAILED = "failed"
    PASS = "pass"
    SUCCESS = "success"
