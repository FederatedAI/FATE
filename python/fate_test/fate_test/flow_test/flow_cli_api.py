import json
import os
import sys
import shutil
import time
import subprocess
import numpy as np
from pathlib import Path

from prettytable import PrettyTable, ORGMODE
from fate_test.flow_test.flow_process import get_dict_from_file, serving_connect


class TestModel(object):
    def __init__(self, data_base_dir, fate_flow_path, component_name, namespace):
        self.conf_path = None
        self.dsl_path = None
        self.job_id = None
        self.model_id = None
        self.model_version = None
        self.guest_party_id = None
        self.host_party_id = None
        self.arbiter_party_id = None
        self.output_path = None
        self.cache_directory = None

        self.data_base_dir = data_base_dir
        self.fate_flow_path = fate_flow_path
        self.component_name = component_name

        self.python_bin = sys.executable or 'python3'

        self.request_api_info_path = f'./logs/{namespace}/cli_exception.log'
        os.makedirs(os.path.dirname(self.request_api_info_path), exist_ok=True)

    def error_log(self, retmsg):
        if retmsg is None:
            return os.path.abspath(self.request_api_info_path)
        with open(self.request_api_info_path, "a") as f:
            f.write(retmsg)

    def submit_job(self, stop=True):
        try:
            subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", "submit_job", "-d", self.dsl_path,
                                     "-c", self.conf_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = subp.communicate()
            stdout = json.loads(stdout.decode("utf-8"))
            if stdout.get('retcode'):
                self.error_log('job submit: {}'.format(stdout.get('retmsg')) + '\n')
            self.job_id = stdout.get("jobId")
            self.model_id = stdout.get("data").get("model_info").get("model_id")
            self.model_version = stdout.get("data").get("model_info").get("model_version")
            if stop:
                return
            return self.query_status()
        except Exception:
            return

    def job_api(self, command):
        if command == 'stop_job':
            self.submit_job()
            time.sleep(5)
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-j", self.job_id],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('job stop: {}'.format(stdout.get('retmsg')) + '\n')
                if self.query_job() == "canceled":
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'job_log':
            log_file_dir = os.path.join(self.output_path, 'job_{}_log'.format(self.job_id))
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-j", self.job_id, "-o",
                                         log_file_dir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('job log: {}'.format(stdout.get('retmsg')) + '\n')
                return stdout.get('retcode')
            except Exception:
                return

        elif command == 'data_view_query':
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-j", self.job_id,
                                         "-r", "guest"],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('data view queue: {}'.format(stdout.get('retmsg')) + '\n')
                if len(stdout.get("data")) == len(list(get_dict_from_file(self.dsl_path)['components'].keys())) - 1:
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'clean_job':
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-j", self.job_id],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('clean job: {}'.format(stdout.get('retmsg')) + '\n')
                subp = subprocess.Popen([self.python_bin,
                                         self.fate_flow_path,
                                         "-f",
                                         "component_metrics",
                                         "-j",
                                         self.job_id,
                                         "-r",
                                         "guest",
                                         "-p",
                                         str(self.guest_party_id[0]),
                                         "-cpn",
                                         'evaluation_0'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                metric, stderr = subp.communicate()
                metric = json.loads(metric.decode("utf-8"))
                if not metric.get('data'):
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'clean_queue':
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('clean queue: {}'.format(stdout.get('retmsg')) + '\n')
                if not self.query_job(queue=True):
                    return stdout.get('retcode')
            except Exception:
                return

    def query_job(self, job_id=None, queue=False):
        if job_id is None:
            job_id = self.job_id
        time.sleep(1)
        try:
            if not queue:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", "query_job", "-j", job_id],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if not stdout.get('retcode'):
                    return stdout.get("data")[0].get("f_status")
                else:
                    self.error_log('query job: {}'.format(stdout.get('retmsg')) + '\n')
            else:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", "query_job", "-j", job_id, "-s",
                                         "waiting"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if not stdout.get('retcode'):
                    return len(stdout.get("data"))
        except Exception:
            return

    def job_config(self, max_iter):
        try:
            subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", "job_config", "-j", self.job_id, "-r",
                                     "guest", "-p", str(self.guest_party_id[0]), "-o", self.output_path],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = subp.communicate()
            stdout = json.loads(stdout.decode("utf-8"))
            if stdout.get('retcode'):
                self.error_log('job config: {}'.format(stdout.get('retmsg')) + '\n')
            job_conf_path = stdout.get('directory') + '/runtime_conf.json'
            job_conf = get_dict_from_file(job_conf_path)
            if max_iter == job_conf['component_parameters']['common'][self.component_name]['max_iter']:
                return stdout.get('retcode')

        except Exception:
            return

    def query_task(self):
        try:
            subp = subprocess.Popen(
                [self.python_bin, self.fate_flow_path, "-f", "query_task", "-j", self.job_id, "-r", "guest",
                 "-p", str(self.guest_party_id[0]), "-cpn", self.component_name],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = subp.communicate()
            stdout = json.loads(stdout.decode("utf-8"))
            if stdout.get('retcode'):
                self.error_log('task query: {}'.format(stdout.get('retmsg')) + '\n')
            status = stdout.get("data")[0].get("f_status")
            if status == "success":
                return stdout.get('retcode')
        except Exception:
            return

    def component_api(self, command, max_iter=None):
        component_output_path = os.path.join(self.output_path, 'job_{}_output_data'.format(self.job_id))
        if command == 'component_output_data':
            try:
                subp = subprocess.Popen(
                    [self.python_bin, self.fate_flow_path, "-f", command, "-j", self.job_id, "-r",
                     "guest", "-p", str(self.guest_party_id[0]), "-cpn", self.component_name, "-o",
                     component_output_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('component output data: {}'.format(stdout.get('retmsg')) + '\n')
                return stdout.get('retcode')
            except Exception:
                return

        elif command == 'component_output_data_table':
            try:
                subp = subprocess.Popen(
                    [self.python_bin, self.fate_flow_path, "-f", command, "-j", self.job_id, "-r",
                     "guest", "-p", str(self.guest_party_id[0]), "-cpn", self.component_name],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('component output data table: {}'.format(stdout.get('retmsg')) + '\n')
                table = {'table_name': stdout.get("data")[0].get("table_name"),
                         'namespace': stdout.get("data")[0].get("namespace")}
                if not self.table_api('table_info', table):
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'component_output_model':
            try:
                subp = subprocess.Popen([self.python_bin,
                                         self.fate_flow_path,
                                         "-f",
                                         command,
                                         "-r",
                                         "guest",
                                         "-j",
                                         self.job_id,
                                         "-p",
                                         str(self.guest_party_id[0]),
                                         "-cpn",
                                         self.component_name],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('component output model: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get("data"):
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'component_parameters':
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-j", self.job_id,
                                         "-r", "guest", "-p", str(self.guest_party_id[0]), "-cpn", self.component_name],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('component parameters: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get('data', {}).get('ComponentParam', {}).get('max_iter', {}) == max_iter:
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'component_metrics':
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-j", self.job_id,
                                         "-r", "guest", "-p", str(self.guest_party_id[0]), "-cpn", 'evaluation_0'],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('component metrics: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get("data"):
                    metrics_file = self.output_path + '{}_metrics.json'.format(self.job_id)
                    with open(metrics_file, 'w') as fp:
                        json.dump(stdout.get("data"), fp)
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'component_metric_all':
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-j", self.job_id,
                                         "-r", "guest", "-p", str(self.guest_party_id[0]), "-cpn", 'evaluation_0'],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('component metric all: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get("data"):
                    metric_all_file = self.output_path + '{}_metric_all.json'.format(self.job_id)
                    with open(metric_all_file, 'w') as fp:
                        json.dump(stdout.get("data"), fp)
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'component_metric_delete':
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-j",
                                         self.job_id, "-r", "guest", "-p", str(self.guest_party_id[0]), "-cpn",
                                         'evaluation_0'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('component metric delete: {}'.format(stdout.get('retmsg')) + '\n')
                subp = subprocess.Popen([self.python_bin,
                                         self.fate_flow_path,
                                         "-f",
                                         "component_metrics",
                                         "-j",
                                         self.job_id,
                                         "-r",
                                         "guest",
                                         "-p",
                                         str(self.guest_party_id[0]),
                                         "-cpn",
                                         'evaluation_0'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                metric, stderr = subp.communicate()
                metric = json.loads(metric.decode("utf-8"))
                if not metric.get('data'):
                    return stdout.get('retcode')
            except Exception:
                return

    def table_api(self, command, table_name):
        if command == 'table_info':
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-t",
                                         table_name['table_name'], "-n", table_name['namespace']],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('table info: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get('data')['namespace'] == table_name['namespace'] and \
                        stdout.get('data')['table_name'] == table_name['table_name']:
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'table_delete':
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-t",
                                         table_name['table_name'], "-n", table_name['namespace']],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('table delete: {}'.format(stdout.get('retmsg')) + '\n')
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", "table_delete", "-t",
                                         table_name['table_name'], "-n", table_name['namespace']],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    return 0
            except Exception:
                return

    def data_upload(self, upload_path, table_index=None):
        upload_file = get_dict_from_file(upload_path)
        upload_file['file'] = str(self.data_base_dir.joinpath(upload_file['file']).resolve())
        upload_file['drop'] = 1
        upload_file['use_local_data'] = 0
        if table_index is not None:
            upload_file['table_name'] = f'{upload_file["file"]}_{table_index}'

        upload_path = self.cache_directory + 'upload_file.json'
        with open(upload_path, 'w') as fp:
            json.dump(upload_file, fp)

        try:
            subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", "upload", "-c",
                                     upload_path, "-drop", "1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = subp.communicate()
            stdout = json.loads(stdout.decode("utf-8"))
            if stdout.get('retcode'):
                self.error_log('data upload: {}'.format(stdout.get('retmsg')) + '\n')
            return self.query_status(stdout.get("jobId"))
        except Exception:
            return

    def data_download(self, table_name, output_path):
        download_config = {
            "table_name": table_name['table_name'],
            "namespace": table_name['namespace'],
            "output_path": output_path + '{}download.csv'.format(self.job_id)
        }
        config_file_path = self.cache_directory + 'download_config.json'
        with open(config_file_path, 'w') as fp:
            json.dump(download_config, fp)
        try:
            subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", "download", "-c", config_file_path],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = subp.communicate()
            stdout = json.loads(stdout.decode("utf-8"))
            if stdout.get('retcode'):
                self.error_log('data download: {}'.format(stdout.get('retmsg')) + '\n')
            return self.query_status(stdout.get("jobId"))
        except Exception:
            return

    def data_upload_history(self, conf_file):
        self.data_upload(conf_file, table_index=1)
        try:
            subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", "upload_history", "-limit", "2"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = subp.communicate()
            stdout = json.loads(stdout.decode("utf-8"))
            if stdout.get('retcode'):
                self.error_log('data upload history: {}'.format(stdout.get('retmsg')) + '\n')
            if len(stdout.get('data')) == 2:
                return stdout.get('retcode')
        except Exception:
            return

    def model_api(self, command, remove_path=None, model_path=None, model_load_conf=None, servings=None):
        if model_load_conf is not None:
            model_load_conf["job_parameters"].update({"model_id": self.model_id,
                                                      "model_version": self.model_version})

        if command == 'load':
            model_load_path = self.cache_directory + 'model_load_file.json'
            with open(model_load_path, 'w') as fp:
                json.dump(model_load_conf, fp)
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-c", model_load_path],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('model load: {}'.format(stdout.get('retmsg')) + '\n')
                return stdout.get('retcode')
            except Exception:
                return

        elif command == 'bind':
            service_id = "".join([str(i) for i in np.random.randint(9, size=8)])
            model_load_conf.update({"service_id": service_id, "servings": [servings]})
            model_bind_path = self.cache_directory + 'model_load_file.json'
            with open(model_bind_path, 'w') as fp:
                json.dump(model_load_conf, fp)
            try:
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-c", model_bind_path],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if stdout.get('retcode'):
                    self.error_log('model bind: {}'.format(stdout.get('retmsg')) + '\n')
                else:
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'import':
            config_data = {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "role": "guest",
                "party_id": self.guest_party_id[0],
                "file": model_path
            }
            config_file_path = self.cache_directory + 'model_import.json'
            with open(config_file_path, 'w') as fp:
                json.dump(config_data, fp)
            try:
                remove_path = Path(remove_path + self.model_version)
                if os.path.isdir(remove_path):
                    shutil.rmtree(remove_path)
                subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-c", config_file_path],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = subp.communicate()
                stdout = json.loads(stdout.decode("utf-8"))
                if not stdout.get('retcode') and os.path.isdir(remove_path):
                    return 0
                else:
                    self.error_log('model import: {}'.format(stdout.get('retmsg')) + '\n')
            except Exception:
                return

        elif command == 'export':
            config_data = {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "role": "guest",
                "party_id": self.guest_party_id[0]
            }
            config_file_path = self.cache_directory + 'model_export.json'
            with open(config_file_path, 'w') as fp:
                json.dump(config_data, fp)
            subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-c", config_file_path, "-o",
                                     self.output_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = subp.communicate()
            stdout = json.loads(stdout.decode("utf-8"))
            if stdout.get('retcode'):
                self.error_log('model export: {}'.format(stdout.get('retmsg')) + '\n')
            else:
                export_model_path = stdout.get('file')
                return stdout.get('retcode'), export_model_path

        elif command in ['store', 'restore']:
            config_data = {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "role": "guest",
                "party_id": self.guest_party_id[0]
            }
            config_file_path = self.cache_directory + 'model_store.json'
            with open(config_file_path, 'w') as fp:
                json.dump(config_data, fp)

            subp = subprocess.Popen([self.python_bin, self.fate_flow_path, "-f", command, "-c", config_file_path],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = subp.communicate()
            stdout = json.loads(stdout.decode("utf-8"))
            if stdout.get('retcode'):
                self.error_log('model {}: {}'.format(command, stdout.get('retmsg')) + '\n')
            return stdout.get('retcode')

    def query_status(self, job_id=None):
        while True:
            time.sleep(5)
            status = self.query_job(job_id=job_id)
            if status and status in ["waiting", "running", "success"]:
                if status and status == "success":
                    return 0
            else:
                return

    def set_config(self, guest_party_id, host_party_id, arbiter_party_id, path, component_name):
        config = get_dict_from_file(path)
        config["initiator"]["party_id"] = guest_party_id[0]
        config["role"]["guest"] = guest_party_id
        config["role"]["host"] = host_party_id
        if "arbiter" in config["role"]:
            config["role"]["arbiter"] = arbiter_party_id
        self.guest_party_id = guest_party_id
        self.host_party_id = host_party_id
        self.arbiter_party_id = arbiter_party_id
        conf_file_path = self.cache_directory + 'conf_file.json'
        with open(conf_file_path, 'w') as fp:
            json.dump(config, fp)
        self.conf_path = conf_file_path
        return config['component_parameters']['common'][component_name]['max_iter']


def judging_state(retcode):
    if not retcode and retcode is not None:
        return 'success'
    else:
        return 'failed'


def run_test_api(config_json, namespace):
    output_path = './output/flow_test_data/'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fate_flow_path = config_json['data_base_dir'] / 'fateflow' / 'python' / 'fate_flow' / 'fate_flow_client.py'
    if not fate_flow_path.exists():
        raise FileNotFoundError(f'fate_flow not found. filepath: {fate_flow_path}')
    test_api = TestModel(config_json['data_base_dir'], str(fate_flow_path), config_json['component_name'], namespace)
    test_api.dsl_path = config_json['train_dsl_path']
    test_api.cache_directory = config_json['cache_directory']
    test_api.output_path = str(os.path.abspath(output_path)) + '/'

    conf_path = config_json['train_conf_path']
    guest_party_id = config_json['guest_party_id']
    host_party_id = config_json['host_party_id']
    arbiter_party_id = config_json['arbiter_party_id']
    upload_file_path = config_json['upload_file_path']
    model_file_path = config_json['model_file_path']
    conf_file = get_dict_from_file(upload_file_path)

    serving_connect_bool = serving_connect(config_json['serving_setting'])
    remove_path = str(config_json['data_base_dir']).split("python")[
        0] + '/model_local_cache/guest#{}#arbiter-{}#guest-{}#host-{}#model/'.format(
        guest_party_id[0], arbiter_party_id[0], guest_party_id[0], host_party_id[0])
    max_iter = test_api.set_config(guest_party_id, host_party_id, arbiter_party_id, conf_path,
                                   config_json['component_name'])

    data = PrettyTable()
    data.set_style(ORGMODE)
    data.field_names = ['data api name', 'status']
    data.add_row(['data upload', judging_state(test_api.data_upload(upload_file_path))])
    data.add_row(['data download', judging_state(test_api.data_download(conf_file, output_path))])
    data.add_row(
        ['data upload history', judging_state(test_api.data_upload_history(upload_file_path))])
    print(data.get_string(title="data api"))

    table = PrettyTable()
    table.set_style(ORGMODE)
    table.field_names = ['table api name', 'status']
    table.add_row(['table info', judging_state(test_api.table_api('table_info', conf_file))])
    table.add_row(['delete table', judging_state(test_api.table_api('table_delete', conf_file))])
    print(table.get_string(title="table api"))

    job = PrettyTable()
    job.set_style(ORGMODE)
    job.field_names = ['job api name', 'status']
    job.add_row(['job stop', judging_state(test_api.job_api('stop_job'))])
    job.add_row(['job submit', judging_state(test_api.submit_job(stop=False))])
    job.add_row(['job query', judging_state(False if test_api.query_job() == "success" else True)])
    job.add_row(['job data view', judging_state(test_api.job_api('data_view_query'))])
    job.add_row(['job config', judging_state(test_api.job_config(max_iter=max_iter))])
    job.add_row(['job log', judging_state(test_api.job_api('job_log'))])

    task = PrettyTable()
    task.set_style(ORGMODE)
    task.field_names = ['task api name', 'status']
    task.add_row(['task query', judging_state(test_api.query_task())])
    print(task.get_string(title="task api"))

    component = PrettyTable()
    component.set_style(ORGMODE)
    component.field_names = ['component api name', 'status']
    component.add_row(['output data', judging_state(test_api.component_api('component_output_data'))])
    component.add_row(['output table', judging_state(test_api.component_api('component_output_data_table'))])
    component.add_row(['output model', judging_state(test_api.component_api('component_output_model'))])
    component.add_row(
        ['component parameters', judging_state(test_api.component_api('component_parameters', max_iter=max_iter))])
    component.add_row(['metrics', judging_state(test_api.component_api('component_metrics'))])
    component.add_row(['metrics all', judging_state(test_api.component_api('component_metric_all'))])

    model = PrettyTable()
    model.set_style(ORGMODE)
    model.field_names = ['model api name', 'status']
    if not config_json.get('component_is_homo') and serving_connect_bool:
        model_load_conf = get_dict_from_file(model_file_path)
        model_load_conf["initiator"]["party_id"] = guest_party_id
        model_load_conf["role"].update(
            {"guest": [guest_party_id], "host": [host_party_id], "arbiter": [arbiter_party_id]})
        model.add_row(['model load', judging_state(test_api.model_api('load', model_load_conf=model_load_conf))])
        model.add_row(['model bind', judging_state(
            test_api.model_api('bind', model_load_conf=model_load_conf, servings=config_json['serving_setting']))])

    status, model_path = test_api.model_api('export')
    model.add_row(['model export', judging_state(status)])
    model.add_row(['model import', (judging_state(
        test_api.model_api('import', remove_path=remove_path, model_path=model_path)))])
    model.add_row(['model store', (judging_state(test_api.model_api('store')))])
    model.add_row(['model restore', (judging_state(test_api.model_api('restore')))])
    print(model.get_string(title="model api"))

    component.add_row(['metrics delete', judging_state(test_api.component_api('component_metric_delete'))])
    print(component.get_string(title="component api"))

    test_api.submit_job()
    test_api.submit_job()
    test_api.submit_job()

    job.add_row(['clean job', judging_state(test_api.job_api('clean_job'))])
    job.add_row(['clean queue', judging_state(test_api.job_api('clean_queue'))])
    print(job.get_string(title="job api"))
    print('Please check the error content: {}'.format(test_api.error_log(None)))
