import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import requests
from contextlib import closing
from prettytable import PrettyTable, ORGMODE
from fate_test.flow_test.flow_process import Base, get_dict_from_file, download_from_request


class TestModel(Base):
    request_api_info_path = './rest_api.log'
    if os.path.exists(request_api_info_path):
        os.remove(request_api_info_path)

    def error_log(self, retmsg):
        with open(self.request_api_info_path, "a") as f:
            f.write(retmsg)

    def submit_job(self, stop=True):
        post_data = {'job_runtime_conf': self.config, 'job_dsl': self.dsl}
        try:
            response = requests.post("/".join([self.server_url, "job", "submit"]), json=post_data)
            if response.status_code == 200:
                if response.json().get('retcode'):
                    self.error_log('job submit: {}'.format(response.json().get('retmsg')) + '\n')
                self.job_id = response.json().get("jobId")
                self.model_id = response.json().get("data").get("model_info").get("model_id")
                self.model_version = response.json().get("data").get("model_info").get("model_version")
                if stop:
                    return
                return self.query_status()
        except Exception:
            return

    def job_dsl_generate(self):
        post_data = {
            'train_dsl': '{"components": {"dataio_0": {"module": "DataIO", "input": {"data": {"data": []}},'
                         '"output": {"data": ["train"], "model": ["hetero_lr"]}}}}',
            'cpn_str': 'dataio_0'
        }
        try:
            response = requests.post("/".join([self.server_url, "job", "dsl/generate"]), json=post_data)
            if response.status_code == 200:
                if response.json().get('retcode'):
                    self.error_log('job dsl generate: {}'.format(response.json().get('retmsg')) + '\n')
                if response.json().get('data')['components']['dataio_0']['input']['model'][
                    0] == 'pipeline.dataio_0.hetero_lr':
                    return response.json().get('retcode')
        except Exception:
            return

    def job_api(self, command, output_path=None):
        post_data = {'job_id': self.job_id}
        if command == 'rerun':
            try:
                response = requests.post("/".join([self.server_url, "job", command]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('job rerun: {}'.format(response.json().get('retmsg')) + '\n')
                    return self.query_status()
            except Exception:
                return

        elif command == 'stop':
            self.submit_job()
            time.sleep(5)
            try:
                response = requests.post("/".join([self.server_url, "job", command]), json={'job_id': self.job_id})
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('job stop: {}'.format(response.json().get('retmsg')) + '\n')
                    if self.query_job() == "canceled":
                        return response.json().get('retcode')
            except Exception:
                return

        elif command == 'data/view/query':
            try:
                response = requests.post("/".join([self.server_url, "job", command]), json=post_data)
                if response.json().get('retcode'):
                    self.error_log('data view query: {}'.format(response.json().get('retmsg')) + '\n')
                if len(response.json().get("data")) == len(self.dsl['components'].keys()) - 1:
                    return response.json().get('retcode')
            except Exception:
                return

        elif command == 'list/job':
            post_data = {'limit': 3}
            try:
                response = requests.post("/".join([self.server_url, "job", "list/job"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('job list: {}'.format(response.json().get('retmsg')) + '\n')
                    if len(response.json().get('data')) == post_data["limit"]:
                        return response.json().get('retcode')
            except Exception:
                return

        elif command == 'log':
            post_data = {'job_id': self.job_id}
            tar_file_name = 'job_{}_log.tar.gz'.format(post_data['job_id'])
            extract_dir = os.path.join(output_path, tar_file_name.replace('.tar.gz', ''))
            with closing(requests.get("/".join([self.server_url, "job", command]),
                                      json=post_data, stream=True)) as response:
                if response.status_code == 200:
                    try:
                        download_from_request(http_response=response, tar_file_name=tar_file_name,
                                              extract_dir=extract_dir)
                        return 0
                    except Exception as e:
                        self.error_log('job log: {}'.format(e) + '\n')
                        return

        elif command == 'clean/queue':
            try:
                response = requests.post("/".join([self.server_url, "job", command]))
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('clean queue: {}'.format(response.json().get('retmsg')) + '\n')
                    if not self.query_job(queue=True):
                        return response.json().get('retcode')
            except Exception:
                return

    def query_job(self, job_id=None, queue=False):
        if job_id is None:
            job_id = self.job_id
        time.sleep(1)
        try:
            if not queue:
                response = requests.post("/".join([self.server_url, "job", "query"]), json={'job_id': job_id})
                if response.status_code == 200 and response.json().get("data"):
                    status = response.json().get("data")[0].get("f_status")
                    return status
                else:
                    self.error_log('query job: {}'.format(response.json().get('retmsg')) + '\n')
            else:
                response = requests.post("/".join([self.server_url, "job", "query"]), json={'status': 'waiting'})
                if response.status_code == 200 and response.json().get("data"):
                    return len(response.json().get("data"))

        except Exception:
            return

    def job_config(self, max_iter, output_path):
        post_data = {
            'job_id': self.job_id,
            "role": "guest",
            "party_id": self.guest_party_id[0],
            "output_path": output_path
        }
        try:
            response = requests.post("/".join([self.server_url, "job", "config"]), json=post_data)
            if response.status_code == 200:
                if response.json().get('retcode'):
                    self.error_log('job config: {}'.format(response.json().get('retmsg')) + '\n')
                job_conf = response.json().get('data')['runtime_conf']
                if max_iter == job_conf['component_parameters']['common'][self.component_name]['max_iter']:
                    return response.json().get('retcode')

        except Exception:
            return

    def query_task(self):
        post_data = {
            'job_id': self.job_id,
            "role": "guest",
            "party_id": self.guest_party_id[0],
            "component_name": self.component_name
        }
        try:
            response = requests.post("/".join([self.server_url, "job", "task/query"]), json=post_data)
            if response.status_code == 200:
                if response.json().get('retcode'):
                    self.error_log('task query: {}'.format(response.json().get('retmsg')) + '\n')
                status = response.json().get("data")[0].get("f_status")
                if status == "success":
                    return response.json().get('retcode')
        except Exception:
            return

    def list_task(self):
        post_data = {'limit': 3}
        try:
            response = requests.post("/".join([self.server_url, "job", "list/task"]), json=post_data)
            if response.status_code == 200:
                if response.json().get('retcode'):
                    self.error_log('list task: {}'.format(response.json().get('retmsg')) + '\n')
                if response.json().get("data") and len(response.json().get('data')) == post_data["limit"]:
                    return response.json().get('retcode')
        except Exception:
            return

    def component_api(self, command, output_path=None, max_iter=None):
        post_data = {
            "job_id": self.job_id,
            "role": "guest",
            "party_id": self.guest_party_id[0],
            "component_name": self.component_name
        }
        if command == 'output/data':
            tar_file_name = 'job_{}_{}_output_data.tar.gz'.format(post_data['job_id'], post_data['component_name'])
            extract_dir = os.path.join(output_path, tar_file_name.replace('.tar.gz', ''))
            with closing(requests.get("/".join([self.server_url, "tracking", "component/output/data/download"]),
                                      json=post_data, stream=True)) as response:
                if response.status_code == 200:
                    try:
                        download_from_request(http_response=response, tar_file_name=tar_file_name,
                                              extract_dir=extract_dir)
                        return 0
                    except Exception as e:
                        self.error_log('component output data: {}'.format(e) + '\n')
                        return

        elif command == 'output/data/table':
            try:
                response = requests.post("/".join([self.server_url, "tracking", "component/output/data/table"]),
                                         json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log(
                            'component output data table: {}'.format(response.json().get('retmsg')) + '\n')
                    table = {'table_name': response.json().get("data")[0].get("table_name"),
                             'namespace': response.json().get("data")[0].get("namespace")}
                    if not self.table_api('table_info', table):
                        return response.json().get('retcode')
            except Exception:
                return

        elif command == 'output/model':
            try:
                response = requests.post("/".join([self.server_url, "tracking", "component/output/model"]),
                                         json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('component output model: {}'.format(response.json().get('retmsg')) + '\n')
                    if response.json().get("data"):
                        return response.json().get('retcode')
            except Exception:
                return

        elif command == 'parameters':
            try:
                response = requests.post("/".join([self.server_url, "tracking", "component/parameters"]),
                                         json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('component parameters: {}'.format(response.json().get('retmsg')) + '\n')
                    if response.json().get("data").get("HeteroLogisticParam").get("max_iter") == max_iter:
                        return response.json().get('retcode')
            except Exception:
                return

        elif command == 'summary/download':
            try:
                response = requests.post("/".join([self.server_url, "tracking", "component/summary/download"]),
                                         json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log(
                            'component summary download: {}'.format(response.json().get('retmsg')) + '\n')
                    if response.json().get("data"):
                        file = output_path + '{}_summary.json'.format(self.job_id)
                        os.makedirs(os.path.dirname(file), exist_ok=True)
                        with open(file, 'w') as fp:
                            json.dump(response.json().get("data"), fp)
                        return response.json().get('retcode')
            except Exception:
                return

    def component_metric(self, command, output_path=None):
        post_data = {
            "job_id": self.job_id,
            "role": "guest",
            "party_id": self.guest_party_id[0],
            "component_name": 'evaluation_0'
        }
        if command == 'metrics':
            try:
                response = requests.post("/".join([self.server_url, "tracking", "component/metrics"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('component metrics: {}'.format(response.json().get('retmsg')) + '\n')
                    if response.json().get("data"):
                        file = output_path + '{}_metrics.json'.format(self.job_id)
                        os.makedirs(os.path.dirname(file), exist_ok=True)
                        with open(file, 'w') as fp:
                            json.dump(response.json().get("data"), fp)
                        return response.json().get('retcode')
            except Exception:
                return

        elif command == 'metric/all':
            try:
                response = requests.post("/".join([self.server_url, "tracking", "component/metric/all"]),
                                         json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('component metric all: {}'.format(response.json().get('retmsg')) + '\n')
                    if response.json().get("data"):
                        file = output_path + '{}_metric_all.json'.format(self.job_id)
                        os.makedirs(os.path.dirname(file), exist_ok=True)
                        with open(file, 'w') as fp:
                            json.dump(response.json().get("data"), fp)
                        return response.json().get('retcode')
            except Exception:
                return

        elif command == 'metric/delete':
            try:
                response = requests.post("/".join([self.server_url, "tracking", "component/metric/delete"]),
                                         json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('component metric delete: {}'.format(response.json().get('retmsg')) + '\n')
                    response = requests.post("/".join([self.server_url, "tracking", "component/metrics"]),
                                             json=post_data)
                    if response.status_code == 200:
                        if not response.json().get("data"):
                            return response.json().get('retcode')
            except Exception:
                return

    def component_list(self):
        post_data = {'job_id': self.job_id}
        try:
            response = requests.post("/".join([self.server_url, "tracking", "component/list"]), json=post_data)
            if response.status_code == 200:
                if response.json().get('retcode'):
                    self.error_log('component list: {}'.format(response.json().get('retmsg')) + '\n')
                if len(response.json().get('data')['components']) == len(list(self.dsl['components'].keys())):
                    return response.json().get('retcode')
        except Exception:
            raise

    def table_api(self, command, table_name):
        post_data = {
            "table_name": table_name['table_name'],
            "namespace": table_name['namespace']
        }
        if command == 'table/info':
            try:
                response = requests.post("/".join([self.server_url, "table", "table_info"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('table info: {}'.format(response.json().get('retmsg')) + '\n')
                    if response.json().get('data')['namespace'] == table_name['namespace'] and \
                            response.json().get('data')['table_name'] == table_name['table_name']:
                        return response.json().get('retcode')

            except Exception:
                return

        elif command == 'table/delete':
            try:
                response = requests.post("/".join([self.server_url, "table", "delete"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('table delete: {}'.format(response.json().get('retmsg')) + '\n')
                    response = requests.post("/".join([self.server_url, "table", "delete"]), json=post_data)
                    if response.status_code == 200 and response.json().get('retcode'):
                        return 0
            except Exception:
                return

    def data_upload(self, post_data, work_mode, table_index=None):
        post_data.update({"drop": 1, "use_local_data": 0, "work_mode": work_mode})
        if table_index is not None:
            post_data.update({"table_name": post_data["file"] + f'_{table_index}'})
        try:
            response = requests.post("/".join([self.server_url, "data", "upload"]), json=post_data)
            if response.status_code == 200:
                if response.json().get('retcode'):
                    self.error_log('data upload: {}'.format(response.json().get('retmsg')) + '\n')
                return self.query_status(response.json().get("jobId"))
        except Exception:
            return

    def data_download(self, table_name, output_path, work_mode):
        post_data = {
            "table_name": table_name['table_name'],
            "namespace": table_name['namespace'],
            "output_path": output_path + '{}download.csv'.format(self.job_id),
            "work_mode": work_mode
        }
        try:
            response = requests.post("/".join([self.server_url, "data", "download"]), json=post_data)
            if response.status_code == 200:
                if response.json().get('retcode'):
                    self.error_log('data download: {}'.format(response.json().get('retmsg')) + '\n')
                return self.query_status(response.json().get("jobId"))
        except Exception:
            return

    def data_upload_history(self, conf_file, work_mode):
        self.data_upload(conf_file, work_mode=work_mode, table_index=1)
        post_data = {"limit": 2}
        try:
            response = requests.post("/".join([self.server_url, "data", "upload/history"]), json=post_data)
            if response.status_code == 200:
                if response.json().get('retcode'):
                    self.error_log('data upload history: {}'.format(response.json().get('retmsg')) + '\n')
                if len(response.json().get('data')) == post_data["limit"]:
                    return response.json().get('retcode')
        except Exception:
            return

    def tag_api(self, command, tag_name=None, new_tag_name=None):
        post_data = {
            "tag_name": tag_name
        }
        if command == 'tag/retrieve':
            try:
                response = requests.post("/".join([self.server_url, "model", "tag/retrieve"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('tag retrieve: {}'.format(response.json().get('retmsg')) + '\n')
                    if not response.json().get('retcode'):
                        return response.json().get('data')['tags'][0]['name']
            except Exception:
                return

        elif command == 'tag/create':
            try:
                response = requests.post("/".join([self.server_url, "model", "tag/create"]), json=post_data)
                if response.status_code == 200:
                    self.error_log('tag create: {}'.format(response.json().get('retmsg')) + '\n')
                    if self.tag_api('tag/retrieve', tag_name=tag_name) == tag_name:
                        return 0
            except Exception:
                return

        elif command == 'tag/destroy':
            try:
                response = requests.post("/".join([self.server_url, "model", "tag/destroy"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('tag destroy: {}'.format(response.json().get('retmsg')) + '\n')
                    if not self.tag_api('tag/retrieve', tag_name=tag_name):
                        return 0
            except Exception:
                return

        elif command == 'tag/update':
            post_data = {
                "tag_name": tag_name,
                "new_tag_name": new_tag_name
            }
            try:
                response = requests.post("/".join([self.server_url, "model", "tag/update"]), json=post_data)
                if response.status_code == 200:
                    self.error_log('tag update: {}'.format(response.json().get('retmsg')) + '\n')
                    if self.tag_api('tag/retrieve', tag_name=new_tag_name) == new_tag_name:
                        return 0
            except Exception:
                return

        elif command == 'tag/list':
            post_data = {"limit": 1}
            try:
                response = requests.post("/".join([self.server_url, "model", "tag/list"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('tag list: {}'.format(response.json().get('retmsg')) + '\n')
                    if len(response.json().get('data')['tags']) == post_data['limit']:
                        return response.json().get('retcode')
            except Exception:
                return

    def model_api(self, command, output_path=None, remove_path=None, model_path=None, arbiter_party_id=None,
                  tag_name=None):
        if command == 'model/load':
            post_data = {
                "job_id": self.job_id
            }
            try:
                response = requests.post("/".join([self.server_url, "model", "load"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('model load: {}'.format(response.json().get('retmsg')) + '\n')
                    return response.json().get('retcode')
            except Exception:
                return

        elif command == 'model/bind':
            post_data = {
                "job_id": self.job_id,
                "service_id": f"auto_test_{datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')}"
            }
            try:
                response = requests.post("/".join([self.server_url, "model", "bind"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('model bind: {}'.format(response.json().get('retmsg')) + '\n')
                    return response.json().get('retcode')
            except Exception:
                return

        elif command == 'model/import':
            config_data = {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "role": "guest",
                "party_id": self.guest_party_id[0],
                "file": model_path
            }
            try:
                remove_path = Path(remove_path + self.model_version)
                if os.path.exists(model_path):
                    files = {'file': open(model_path, 'rb')}
                else:
                    return
                if os.path.isdir(remove_path):
                    shutil.rmtree(remove_path)
                response = requests.post("/".join([self.server_url, "model", "import"]), data=config_data, files=files)
                if response.status_code == 200:
                    if os.path.isdir(remove_path):
                        return 0
            except Exception:
                return

        elif command == 'model/export':
            post_data = {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "role": "guest",
                "party_id": self.guest_party_id[0],
            }
            tar_file_name = '{}_{}_model_export.zip'.format(post_data['model_id'], post_data['model_version'])
            archive_file_path = os.path.join(output_path, tar_file_name)
            with closing(requests.get("/".join([self.server_url, "model", "export"]), json=post_data,
                                      stream=True)) as response:
                if response.status_code == 200:
                    try:
                        with open(archive_file_path, 'wb') as fw:
                            for chunk in response.iter_content(1024):
                                if chunk:
                                    fw.write(chunk)
                    except Exception:
                        return
                    return 0, archive_file_path

        elif command == 'model/migrate':
            post_data = {
                "job_parameters": {
                    "federated_mode": "MULTIPLE"
                },
                "migrate_initiator": {
                    "role": "guest",
                    "party_id": self.guest_party_id[0]
                },
                "role": {
                    "guest": self.guest_party_id,
                    "arbiter": arbiter_party_id,
                    "host": self.host_party_id
                },
                "migrate_role": {
                    "guest": self.guest_party_id,
                    "arbiter": arbiter_party_id,
                    "host": self.host_party_id
                },
                "execute_party": {
                    "guest": self.guest_party_id,
                    "arbiter": arbiter_party_id,
                    "host": self.host_party_id
                },
                "model_id": self.model_id,
                "model_version": self.model_version,
                "unify_model_version": self.job_id + '_01'
            }
            try:
                response = requests.post("/".join([self.server_url, "model", "migrate"]), json=post_data)
                if response.status_code == 200:
                    self.error_log('model migrate: {}'.format(response.json().get('retmsg')) + '\n')
                return response.json().get("retcode")
            except Exception:
                return

        elif command == 'model_tag/create':
            post_data = {
                "job_id": self.job_id,
                "tag_name": tag_name
            }
            try:
                response = requests.post("/".join([self.server_url, "model", "model_tag/create"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('model tag create: {}'.format(response.json().get('retmsg')) + '\n')
                    if self.model_api('model_tag/retrieve')[0].get('name') == post_data['tag_name']:
                        return 0
            except Exception:
                return

        elif command == 'model_tag/remove':
            post_data = {
                "job_id": self.job_id,
                "tag_name": tag_name
            }
            try:
                response = requests.post("/".join([self.server_url, "model", "model_tag/remove"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('model tag remove: {}'.format(response.json().get('retmsg')) + '\n')
                    if not len(self.model_api('model_tag/retrieve')):
                        return 0
            except Exception:
                return

        elif command == 'model_tag/retrieve':
            post_data = {
                "job_id": self.job_id
            }
            try:
                response = requests.post("/".join([self.server_url, "model", "model_tag/retrieve"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('model tag retrieve: {}'.format(response.json().get('retmsg')) + '\n')
                    return response.json().get('data')['tags']
            except Exception:
                return

        elif command == 'model/deploy':
            post_data = {
                "model_id": self.model_id,
                "model_version": self.model_version
            }
            try:
                response = requests.post("/".join([self.server_url, "model", "deploy"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('model deploy: {}'.format(response.json().get('retmsg')) + '\n')
                    if response.json().get('data')['model_id'] == self.model_id and \
                            response.json().get('data')['model_version'] != self.model_version:
                        self.model_id = response.json().get('data')['model_id']
                        self.model_version = response.json().get('data')['model_version']
                        self.job_id = response.json().get('data')['model_version']
                    return response.json().get('retcode')
            except Exception:
                return

        elif command == 'model/conf':
            post_data = {
                "model_id": self.model_id,
                "model_version": self.model_version
            }
            try:
                response = requests.post("/".join([self.server_url, "model", "get/predict/conf"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('model conf: {}'.format(response.json().get('retmsg')) + '\n')
                    if response.json().get('data'):
                        if response.json().get('data')['job_parameters']['common']['model_id'] == post_data['model_id']\
                                and response.json().get('data')['job_parameters']['common']['model_version'] == \
                                post_data['model_version'] and response.json().get('data')['initiator']['party_id'] == \
                                self.guest_party_id[0] and response.json().get('data')['initiator']['role'] == 'guest':
                            return response.json().get('retcode')

            except Exception:
                return

        elif command == 'model/dsl':
            post_data = {
                "model_id": self.model_id,
                "model_version": self.model_version
            }
            try:
                response = requests.post("/".join([self.server_url, "model", "get/predict/dsl"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('model dsl: {}'.format(response.json().get('retmsg')) + '\n')
                    model_dsl_cpn = list(response.json().get('data')['components'].keys())
                    train_dsl_cpn = list(self.dsl['components'].keys())
                    if len([k for k in model_dsl_cpn if k in train_dsl_cpn]) == len(train_dsl_cpn):
                        return response.json().get('retcode')
            except Exception:
                return

        elif command == 'model/query':
            post_data = {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "role": "guest",
                "party_id": self.guest_party_id[0]
            }
            try:
                response = requests.post("/".join([self.server_url, "model", "query"]), json=post_data)
                if response.status_code == 200:
                    if response.json().get('retcode'):
                        self.error_log('model query: {}'.format(response.json().get('retmsg')) + '\n')
                    if response.json().get('data')[0].get('f_model_id') == post_data['model_id'] and \
                            response.json().get('data')[0].get('f_model_version') == post_data['model_version'] and \
                            response.json().get('data')[0].get('f_role') == post_data['role'] and \
                            response.json().get('data')[0].get('f_party_id') == str(post_data['party_id']):
                        return response.json().get('retcode')
            except Exception:
                return

    def query_status(self, job_id=None):
        while True:
            time.sleep(5)
            status = self.query_job(job_id=job_id)
            if status and status in ["waiting", "running", "success"]:
                if status and status == "success":
                    return 0
            else:
                return


def judging_state(retcode):
    if not retcode and retcode is not None:
        return 'success'
    else:
        return 'failed'


def run_test_api(config_json):
    output_path = './output/flow_test_data/'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_path = str(os.path.abspath(output_path)) + '/'
    guest_party_id = config_json['guest_party_id']
    host_party_id = config_json['host_party_id']
    arbiter_party_id = config_json['arbiter_party_id']
    train_conf_path = config_json['train_conf_path']
    train_dsl_path = config_json['train_dsl_path']
    upload_file_path = config_json['upload_file_path']
    work_mode = config_json['work_mode']
    remove_path = str(config_json[
                          'data_base_dir']) + '/model_local_cache/guest#{}#arbiter-{}#guest-{}#host-{}#model/'.format(
        guest_party_id[0], arbiter_party_id[0], guest_party_id[0], host_party_id[0])

    test_api = TestModel(config_json['server_url'], component_name=config_json['component_name'])
    job_conf = test_api.set_config(guest_party_id, host_party_id, arbiter_party_id, train_conf_path, work_mode)
    max_iter = job_conf['component_parameters']['common']['hetero_lr_0']['max_iter']
    test_api.set_dsl(train_dsl_path)
    conf_file = get_dict_from_file(upload_file_path)

    data = PrettyTable()
    data.set_style(ORGMODE)
    data.field_names = ['data api name', 'status']
    data.add_row(['data upload', judging_state(test_api.data_upload(conf_file, work_mode=work_mode))])
    data.add_row(['data download', judging_state(test_api.data_download(conf_file, output_path, work_mode))])
    data.add_row(['data upload history', judging_state(test_api.data_upload_history(conf_file, work_mode=work_mode))])
    print(data.get_string(title="data api"))

    table = PrettyTable()
    table.set_style(ORGMODE)
    table.field_names = ['table api name', 'status']
    table.add_row(['table info', judging_state(test_api.table_api('table/info', conf_file))])
    table.add_row(['delete table', judging_state(test_api.table_api('table/delete', conf_file))])
    print(table.get_string(title="table api"))

    job = PrettyTable()
    job.set_style(ORGMODE)
    job.field_names = ['job api name', 'status']
    job.add_row(['job stop', judging_state(test_api.job_api('stop'))])
    job.add_row(['job rerun', judging_state(test_api.job_api('rerun'))])
    job.add_row(['job submit', judging_state(test_api.submit_job(stop=False))])
    job.add_row(['job query', judging_state(False if test_api.query_job() == "success" else True)])
    job.add_row(['job data view', judging_state(test_api.job_api('data/view/query'))])
    job.add_row(['job list', judging_state(test_api.job_api('list/job'))])
    job.add_row(['job config', judging_state(test_api.job_config(max_iter=max_iter, output_path=output_path))])
    job.add_row(['job log', judging_state(test_api.job_api('log', output_path))])
    job.add_row(['job dsl generate', judging_state(test_api.job_dsl_generate())])
    print(job.get_string(title="job api"))

    task = PrettyTable()
    task.set_style(ORGMODE)
    task.field_names = ['task api name', 'status']
    task.add_row(['task list', judging_state(test_api.list_task())])
    task.add_row(['task query', judging_state(test_api.query_task())])
    print(task.get_string(title="task api"))

    tag = PrettyTable()
    tag.set_style(ORGMODE)
    tag.field_names = ['tag api name', 'status']
    tag.add_row(['create tag', judging_state(test_api.tag_api('tag/create', 'create_job_tag'))])
    tag.add_row(['update tag', judging_state(test_api.tag_api('tag/update', 'create_job_tag', 'update_job_tag'))])
    tag.add_row(['list tag', judging_state(test_api.tag_api('tag/list'))])
    tag.add_row(
        ['retrieve tag', judging_state(not test_api.tag_api('tag/retrieve', 'update_job_tag') == 'update_job_tag')])
    tag.add_row(['destroy tag', judging_state(test_api.tag_api('tag/destroy', 'update_job_tag'))])
    print(tag.get_string(title="tag api"))

    component = PrettyTable()
    component.set_style(ORGMODE)
    component.field_names = ['component api name', 'status']
    component.add_row(['output data', judging_state(test_api.component_api('output/data', output_path=output_path))])
    component.add_row(['output table', judging_state(test_api.component_api('output/data/table'))])
    component.add_row(['output model', judging_state(test_api.component_api('output/model'))])
    component.add_row(['component parameters', judging_state(test_api.component_api('parameters', max_iter=max_iter))])
    component.add_row(
        ['component summary', judging_state(test_api.component_api('summary/download', output_path=output_path))])
    component.add_row(['component list', judging_state(test_api.component_list())])
    component.add_row(['metrics', judging_state(
        test_api.component_metric('metrics', output_path=output_path))])
    component.add_row(['metrics all', judging_state(
        test_api.component_metric('metric/all', output_path=output_path))])

    model = PrettyTable()
    model.set_style(ORGMODE)
    model.field_names = ['model api name', 'status']
    model.add_row(['model load', judging_state(test_api.model_api('model/load'))])
    model.add_row(['model bind', judging_state(test_api.model_api('model/bind'))])
    status, model_path = test_api.model_api('model/export', output_path=output_path)
    model.add_row(['model export', judging_state(status)])
    model.add_row(['model  import', (judging_state(
        test_api.model_api('model/import', remove_path=remove_path, model_path=model_path)))])
    model.add_row(
        ['model_tag create', judging_state(test_api.model_api('model_tag/create', tag_name='model_tag_create'))])
    model.add_row(
        ['model_tag remove', judging_state(test_api.model_api('model_tag/remove', tag_name='model_tag_create'))])
    model.add_row(['model_tag retrieve', judging_state(len(test_api.model_api('model_tag/retrieve')))])
    model.add_row(
        ['model migrate', judging_state(test_api.model_api('model/migrate', arbiter_party_id=arbiter_party_id))])
    model.add_row(['model query', judging_state(test_api.model_api('model/query'))])
    model.add_row(['model deploy', judging_state(test_api.model_api('model/deploy'))])
    model.add_row(['model conf', judging_state(test_api.model_api('model/conf'))])
    model.add_row(['model dsl', judging_state(test_api.model_api('model/dsl'))])
    print(model.get_string(title="model api"))
    component.add_row(['metrics delete', judging_state(
        test_api.component_metric('metric/delete', output_path=output_path))])
    print(component.get_string(title="component api"))

    queue = PrettyTable()
    queue.set_style(ORGMODE)
    queue.field_names = ['api name', 'status']
    test_api.submit_job()
    test_api.submit_job()
    test_api.submit_job()
    queue.add_row(['clean/queue', judging_state(test_api.job_api('clean/queue'))])
    print(queue.get_string(title="queue job"))
