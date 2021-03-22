import json
import os
import shutil
import time
from pathlib import Path

from flow_sdk.client import FlowClient
from prettytable import PrettyTable, ORGMODE
from fate_test.flow_test.flow_process import get_dict_from_file


class TestModel(object):
    def __init__(self, server_url, component_name):
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
        self.component_name = component_name
        self.client = FlowClient(server_url.split(':')[0], server_url.split(':')[1].split('/')[0],
                                 server_url.split(':')[1].split('/')[1])
        self.request_api_info_path = './sdk_api.log'
        if os.path.exists(self.request_api_info_path):
            os.remove(self.request_api_info_path)

    def error_log(self, retmsg):
        with open(self.request_api_info_path, "a") as f:
            f.write(retmsg)

    def submit_job(self, stop=True):
        try:
            stdout = self.client.job.submit(conf_path=self.conf_path, dsl_path=self.dsl_path)
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

    def job_dsl_generate(self):
        train_dsl = {"components": {"dataio_0": {"module": "DataIO", "input": {"data": {"data": []}},
                                                 "output": {"data": ["train"], "model": ["hetero_lr"]}}}}
        train_dsl_path = self.cache_directory + 'generate_dsl_file.json'
        with open(train_dsl_path, 'w') as fp:
            json.dump(train_dsl, fp)
        try:
            stdout = self.client.job.generate_dsl(train_dsl_path=train_dsl_path, cpn_list=['dataio_0'])
            if stdout.get('retcode'):
                self.error_log('job dsl generate: {}'.format(stdout.get('retmsg')) + '\n')
            if stdout.get('data')['components']['dataio_0']['input']['model'][
                0] == 'pipeline.dataio_0.hetero_lr':
                return stdout.get('retcode')
        except Exception:
            return

    def job_api(self, command):
        if command == 'stop':
            self.submit_job()
            time.sleep(5)
            try:
                stdout = self.client.job.stop(job_id=self.job_id)
                if stdout.get('retcode'):
                    self.error_log('job stop: {}'.format(stdout.get('retmsg')) + '\n')
                if self.query_job() == "canceled":
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'list/job':
            try:
                stdout = self.client.job.list(limit=3)
                if stdout.get('retcode'):
                    self.error_log('job list: {}'.format(stdout.get('retmsg')) + '\n')
                if len(stdout.get('data')) == 3:
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'view':
            try:
                stdout = self.client.job.view(job_id=self.job_id)
                if stdout.get('retcode'):
                    self.error_log('job view: {}'.format(stdout.get('retmsg')) + '\n')
                if len(stdout.get("data")) == len(list(get_dict_from_file(self.dsl_path)['components'].keys())) - 1:
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'log':
            log_file_dir = os.path.join(self.output_path, 'job_{}_log'.format(self.job_id))
            try:
                stdout = self.client.job.log(job_id=self.job_id, output_path=log_file_dir)
                if stdout.get('retcode'):
                    self.error_log('job log: {}'.format(stdout.get('retmsg')) + '\n')
                return stdout.get('retcode')
            except Exception:
                return

        elif command == 'clean/queue':
            try:
                stdout = self.client.queue.clean()
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
                stdout = self.client.job.query(job_id=job_id)
                if not stdout.get('retcode'):
                    return stdout.get("data")[0].get("f_status")
                else:
                    self.error_log('query job: {}'.format(stdout.get('retmsg')) + '\n')
            else:
                stdout = self.client.job.query(job_id=job_id, status='waiting')
                if not stdout.get('retcode'):
                    return len(stdout.get("data"))
        except Exception:
            return

    def job_config(self, max_iter):
        try:
            stdout = self.client.job.config(job_id=self.job_id, role="guest", party_id=self.guest_party_id[0],
                                            output_path=self.output_path)
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
            stdout = self.client.task.query(job_id=self.job_id, role="guest", party_id=self.guest_party_id[0],
                                            component_name=self.component_name)
            if stdout.get('retcode'):
                self.error_log('task query: {}'.format(stdout.get('retmsg')) + '\n')
            status = stdout.get("data")[0].get("f_status")
            if status == "success":
                return stdout.get('retcode')
        except Exception:
            return

    def list_task(self):
        try:
            stdout = self.client.task.list(limit=3)
            if stdout.get('retcode'):
                self.error_log('list task: {}'.format(stdout.get('retmsg')) + '\n')
            if stdout.get("data") and len(stdout.get('data')) == 3:
                return stdout.get('retcode')
        except Exception:
            return

    def component_api(self, command, max_iter=None):
        component_output_path = os.path.join(self.output_path, 'job_{}_output_data'.format(self.job_id))
        if command == 'output/data':
            try:
                stdout = self.client.component.output_data(job_id=self.job_id, role="guest",
                                                           party_id=self.guest_party_id[0],
                                                           component_name=self.component_name,
                                                           output_path=component_output_path)
                if stdout.get('retcode'):
                    self.error_log('component output data: {}'.format(stdout.get('retmsg')) + '\n')
                return stdout.get('retcode')
            except Exception:
                return

        elif command == 'output/data/table':
            try:
                stdout = self.client.component.output_data_table(job_id=self.job_id, role="guest",
                                                                 party_id=self.guest_party_id[0],
                                                                 component_name=self.component_name)
                if stdout.get('retcode'):
                    self.error_log('component output data table: {}'.format(stdout.get('retmsg')) + '\n')
                table = {'table_name': stdout.get("data")[0].get("table_name"),
                         'namespace': stdout.get("data")[0].get("namespace")}
                if not self.table_api('table_info', table):
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'output/model':
            try:
                stdout = self.client.component.output_model(job_id=self.job_id, role="guest",
                                                            party_id=self.guest_party_id[0],
                                                            component_name=self.component_name)
                if stdout.get('retcode'):
                    self.error_log('component output model: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get("data"):
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'parameters':
            try:
                stdout = self.client.component.parameters(job_id=self.job_id, role="guest",
                                                          party_id=self.guest_party_id[0],
                                                          component_name=self.component_name)
                if stdout.get('retcode'):
                    self.error_log('component parameters: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get("data").get("HeteroLogisticParam").get("max_iter") == max_iter:
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'summary':
            try:
                stdout = self.client.component.get_summary(job_id=self.job_id, role="guest",
                                                           party_id=self.guest_party_id[0],
                                                           component_name=self.component_name)
                if stdout.get('retcode'):
                    self.error_log('component summary download: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get("data"):
                    summary_file = self.output_path + '{}_summary.json'.format(self.job_id)
                    with open(summary_file, 'w') as fp:
                        json.dump(stdout.get("data"), fp)
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'metrics':
            try:
                stdout = self.client.component.metrics(job_id=self.job_id, role="guest",
                                                       party_id=self.guest_party_id[0],
                                                       component_name='evaluation_0')
                if stdout.get('retcode'):
                    self.error_log('component metrics: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get("data"):
                    metrics_file = self.output_path + '{}_metrics.json'.format(self.job_id)
                    with open(metrics_file, 'w') as fp:
                        json.dump(stdout.get("data"), fp)
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'metric/all':
            try:
                stdout = self.client.component.metric_all(job_id=self.job_id, role="guest",
                                                          party_id=self.guest_party_id[0],
                                                          component_name='evaluation_0')
                if stdout.get('retcode'):
                    self.error_log('component metric all: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get("data"):
                    metric_all_file = self.output_path + '{}_metric_all.json'.format(self.job_id)
                    with open(metric_all_file, 'w') as fp:
                        json.dump(stdout.get("data"), fp)
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'metric/delete':
            try:
                stdout = self.client.component.metric_delete(job_id=self.job_id, date=str(time.strftime("%Y%m%d")))
                if stdout.get('retcode'):
                    self.error_log('component metric delete: {}'.format(stdout.get('retmsg')) + '\n')
                metric = self.client.component.metrics(job_id=self.job_id, role="guest",
                                                       party_id=self.guest_party_id[0],
                                                       component_name='evaluation_0')
                if not metric.get('data'):
                    return stdout.get('retcode')
            except Exception:
                return

    def component_list(self):
        try:
            stdout = self.client.component.list(job_id=self.job_id)
            if stdout.get('retcode'):
                self.error_log('component list: {}'.format(stdout.get('retmsg')) + '\n')
            dsl_json = get_dict_from_file(self.dsl_path)
            if len(stdout.get('data')['components']) == len(list(dsl_json['components'].keys())):
                return stdout.get('retcode')
        except Exception:
            raise

    def table_api(self, command, table_name):
        if command == 'table/info':
            try:
                stdout = self.client.table.info(table_name=table_name['table_name'], namespace=table_name['namespace'])
                if stdout.get('retcode'):
                    self.error_log('table info: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get('data')['namespace'] == table_name['namespace'] and \
                        stdout.get('data')['table_name'] == table_name['table_name']:
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'table/delete':
            try:
                stdout = self.client.table.delete(table_name=table_name['table_name'],
                                                  namespace=table_name['namespace'])

                if stdout.get('retcode'):
                    self.error_log('table delete: {}'.format(stdout.get('retmsg')) + '\n')
                stdout = self.client.table.delete(table_name=table_name['table_name'],
                                                  namespace=table_name['namespace'])
                if stdout.get('retcode'):
                    return 0
            except Exception:
                return

    def data_upload(self, upload_path, work_mode, table_index=None):
        upload_file = get_dict_from_file(upload_path)
        upload_file.update({"use_local_data": 0, "work_mode": work_mode})
        if table_index is not None:
            upload_file.update({"table_name": upload_file["file"] + f'_{table_index}'})
        upload_path = self.cache_directory + 'upload_file.json'
        with open(upload_path, 'w') as fp:
            json.dump(upload_file, fp)
        try:
            stdout = self.client.data.upload(conf_path=upload_path, drop=1)
            if stdout.get('retcode'):
                self.error_log('data upload: {}'.format(stdout.get('retmsg')) + '\n')
            return self.query_status(stdout.get("jobId"))
        except Exception:
            return

    def data_download(self, table_name, output_path, work_mode):
        download_config = {
            "table_name": table_name['table_name'],
            "namespace": table_name['namespace'],
            "output_path": output_path + '{}download.csv'.format(self.job_id),
            "work_mode": work_mode
        }
        config_file_path = self.cache_directory + 'download_config.json'
        with open(config_file_path, 'w') as fp:
            json.dump(download_config, fp)
        try:
            stdout = self.client.data.download(conf_path=config_file_path)
            if stdout.get('retcode'):
                self.error_log('data download: {}'.format(stdout.get('retmsg')) + '\n')
            return self.query_status(stdout.get("jobId"))
        except Exception:
            return

    def data_upload_history(self, conf_file, work_mode):
        self.data_upload(conf_file, work_mode=work_mode, table_index=1)
        try:
            stdout = self.client.data.upload_history(limit=2)
            if stdout.get('retcode'):
                self.error_log('data upload history: {}'.format(stdout.get('retmsg')) + '\n')
            if len(stdout.get('data')) == 2:
                return stdout.get('retcode')
        except Exception:
            return

    def tag_api(self, command, tag_name=None, new_tag_name=None):
        if command == 'tag/query':
            try:
                stdout = self.client.tag.query(tag_name=tag_name)
                if stdout.get('retcode'):
                    self.error_log('tag query: {}'.format(stdout.get('retmsg')) + '\n')
                if not stdout.get('retcode'):
                    return stdout.get('data')['tags'][0]['name']
            except Exception:
                return

        elif command == 'tag/create':
            try:
                stdout = self.client.tag.create(tag_name=tag_name)
                self.error_log('tag create: {}'.format(stdout.get('retmsg')) + '\n')
                if self.tag_api('tag/query', tag_name=tag_name) == tag_name:
                    return 0
            except Exception:
                return

        elif command == 'tag/delete':
            try:
                stdout = self.client.tag.delete(tag_name=tag_name)
                if stdout.get('retcode'):
                    self.error_log('tag delete: {}'.format(stdout.get('retmsg')) + '\n')
                if not self.tag_api('tag/query', tag_name=tag_name):
                    return 0
            except Exception:
                return

        elif command == 'tag/update':
            try:
                stdout = self.client.tag.update(tag_name=tag_name, new_tag_name=new_tag_name)
                self.error_log('tag update: {}'.format(stdout.get('retmsg')) + '\n')
                if self.tag_api('tag/query', tag_name=new_tag_name) == new_tag_name:
                    return 0
            except Exception:
                return

        elif command == 'tag/list':
            try:
                stdout = self.client.tag.list(limit=1)
                if stdout.get('retcode'):
                    self.error_log('tag list: {}'.format(stdout.get('retmsg')) + '\n')
                if len(stdout.get('data')['tags']) == 1:
                    return stdout.get('retcode')
            except Exception:
                return

    def model_api(self, command, remove_path=None, model_path=None, tag_name=None, load_path=None, bind_path=None,
                  remove=False):
        if command == 'model/load':
            try:
                stdout = self.client.model.load(conf_path=load_path)
                if stdout.get('retcode'):
                    self.error_log('model load: {}'.format(stdout.get('retmsg')) + '\n')
                return stdout.get('retcode')
            except Exception:
                return

        elif command == 'model/bind':
            try:
                stdout = self.client.model.bind(conf_path=bind_path)
                if stdout.get('retcode'):
                    self.error_log('model bind: {}'.format(stdout.get('retmsg')) + '\n')
                else:
                    return stdout.get('retcode')
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
            config_file_path = self.cache_directory + 'model_import.json'
            with open(config_file_path, 'w') as fp:
                json.dump(config_data, fp)
            try:
                remove_path = Path(remove_path + self.model_version)
                if os.path.isdir(remove_path):
                    shutil.rmtree(remove_path)
                stdout = self.client.model.import_model(conf_path=config_file_path)
                if not stdout.get('retcode') and os.path.isdir(remove_path):
                    return 0
                else:
                    self.error_log('model import: {}'.format(stdout.get('retmsg')) + '\n')
            except Exception:
                return

        elif command == 'model/export':
            config_data = {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "role": "guest",
                "party_id": self.guest_party_id[0],
                "output_path": self.output_path
            }
            config_file_path = self.cache_directory + 'model_export.json'
            with open(config_file_path, 'w') as fp:
                json.dump(config_data, fp)
            stdout = self.client.model.export_model(conf_path=config_file_path)
            if stdout.get('retcode'):
                self.error_log('model export: {}'.format(stdout.get('retmsg')) + '\n')
            else:
                export_model_path = stdout.get('file')
                return stdout.get('retcode'), export_model_path

        elif command == 'model/migrate':
            config_data = {
                "job_parameters": {
                    "federated_mode": "MULTIPLE"
                },
                "migrate_initiator": {
                    "role": "guest",
                    "party_id": self.guest_party_id[0]
                },
                "role": {
                    "guest": self.guest_party_id,
                    "arbiter": self.arbiter_party_id,
                    "host": self.host_party_id
                },
                "migrate_role": {
                    "guest": self.guest_party_id,
                    "arbiter": self.arbiter_party_id,
                    "host": self.host_party_id
                },
                "execute_party": {
                    "guest": self.guest_party_id,
                    "arbiter": self.arbiter_party_id,
                    "host": self.host_party_id
                },
                "model_id": self.model_id,
                "model_version": self.model_version,
                "unify_model_version": self.job_id + '_01'
            }
            config_file_path = self.cache_directory + 'model_migrate.json'
            with open(config_file_path, 'w') as fp:
                json.dump(config_data, fp)
            try:
                stdout = self.client.model.migrate(conf_path=config_file_path)
                if stdout.get('retcode'):
                    self.error_log('model migrate: {}'.format(stdout.get('retmsg')) + '\n')
                return stdout.get('retcode')
            except Exception:
                return

        elif command == 'model_tag/model':
            try:
                stdout = self.client.model.tag_model(job_id=self.job_id, tag_name=tag_name, remove=remove)
                if stdout.get('retcode'):
                    self.error_log('model tag model: {}'.format(stdout.get('retmsg')) + '\n')
                return self.model_api('model_tag/list', tag_name=tag_name)
            except Exception:
                return

        elif command == 'model_tag/list':
            try:
                stdout = self.client.model.tag_list(job_id=self.job_id)
                if stdout.get('retcode'):
                    self.error_log('model tag retrieve: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get('data')['tags'][0].get('name') == tag_name:
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'model/deploy':
            try:
                stdout = self.client.model.deploy(model_id=self.model_id, model_version=self.model_version)
                if stdout.get('retcode'):
                    self.error_log('model deploy: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get('data')['model_id'] == self.model_id and\
                        stdout.get('data')['model_version'] != self.model_version:
                    self.model_id = stdout.get('data')['model_id']
                    self.model_version = stdout.get('data')['model_version']
                    self.job_id = stdout.get('data')['model_version']
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'model/conf':
            try:
                stdout = self.client.model.get_predict_conf(model_id=self.model_id, model_version=self.model_version)
                if stdout.get('retcode'):
                    self.error_log('model conf: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get('data'):
                    if stdout.get('data')['job_parameters']['common']['model_id'] == self.model_id \
                            and stdout.get('data')['job_parameters']['common']['model_version'] == \
                            self.model_version and stdout.get('data')['initiator']['party_id'] == \
                            self.guest_party_id[0] and stdout.get('data')['initiator']['role'] == 'guest':
                        return stdout.get('retcode')
            except Exception:
                return

        elif command == 'model/dsl':
            try:
                stdout = self.client.model.get_predict_dsl(model_id=self.model_id, model_version=self.model_version)
                if stdout.get('retcode'):
                    self.error_log('model dsl: {}'.format(stdout.get('retmsg')) + '\n')
                model_dsl_cpn = list(stdout.get('data')['components'].keys())
                train_dsl_cpn = list(get_dict_from_file(self.dsl_path)['components'].keys())
                if len([k for k in model_dsl_cpn if k in train_dsl_cpn]) == len(train_dsl_cpn):
                    return stdout.get('retcode')
            except Exception:
                return

        elif command == 'model/query':
            try:
                stdout = self.client.model.get_model_info(model_id=self.model_id, model_version=self.model_version,
                                                          role="guest", party_id=self.guest_party_id[0])
                if stdout.get('retcode'):
                    self.error_log('model query: {}'.format(stdout.get('retmsg')) + '\n')
                if stdout.get('data')[0].get('f_model_id') == self.model_id and \
                        stdout.get('data')[0].get('f_model_version') == self.model_version and \
                        stdout.get('data')[0].get('f_role') == "guest" and \
                        stdout.get('data')[0].get('f_party_id') == str(self.guest_party_id[0]):
                    return stdout.get('retcode')
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

    def set_config(self, guest_party_id, host_party_id, arbiter_party_id, path, work_mode):
        config = get_dict_from_file(path)
        config["initiator"]["party_id"] = guest_party_id[0]
        config["role"]["guest"] = guest_party_id
        config["role"]["host"] = host_party_id
        if config["job_parameters"].get("common"):
            config["job_parameters"]["common"]["work_mode"] = work_mode
        else:
            config["job_parameters"]["work_mode"] = work_mode
        if "arbiter" in config["role"]:
            config["role"]["arbiter"] = arbiter_party_id
        self.guest_party_id = guest_party_id
        self.host_party_id = host_party_id
        self.arbiter_party_id = arbiter_party_id
        hetero_conf_file_path = self.cache_directory + 'hetero_conf_file.json'
        with open(hetero_conf_file_path, 'w') as fp:
            json.dump(config, fp)
        self.conf_path = hetero_conf_file_path
        return config['component_parameters']['common']['hetero_lr_0']['max_iter']


def judging_state(retcode):
    if not retcode and retcode is not None:
        return 'success'
    else:
        return 'failed'


def run_test_api(config_json):
    output_path = './output/flow_test_data/'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_api = TestModel(config_json['server_url'].split('//')[1], config_json['component_name'])
    test_api.dsl_path = config_json['train_dsl_path']
    test_api.cache_directory = config_json['cache_directory']
    test_api.output_path = str(os.path.abspath(output_path)) + '/'
    conf_path = config_json['train_conf_path']
    guest_party_id = config_json['guest_party_id']
    host_party_id = config_json['host_party_id']
    arbiter_party_id = config_json['arbiter_party_id']
    upload_file_path = config_json['upload_file_path']
    conf_file = get_dict_from_file(upload_file_path)
    work_mode = config_json['work_mode']
    remove_path = str(config_json[
                          'data_base_dir']) + '/model_local_cache/guest#{}#arbiter-{}#guest-{}#host-{}#model/'.format(
        guest_party_id[0], arbiter_party_id[0], guest_party_id[0], host_party_id[0])
    max_iter = test_api.set_config(guest_party_id, host_party_id, arbiter_party_id, conf_path, work_mode)

    data = PrettyTable()
    data.set_style(ORGMODE)
    data.field_names = ['data api name', 'status']
    data.add_row(['data upload', judging_state(test_api.data_upload(upload_file_path, work_mode=work_mode))])
    data.add_row(['data download', judging_state(test_api.data_download(conf_file, output_path, work_mode))])
    data.add_row(
        ['data upload history', judging_state(test_api.data_upload_history(upload_file_path, work_mode=work_mode))])
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
    job.add_row(['job submit', judging_state(test_api.submit_job(stop=False))])
    job.add_row(['job query', judging_state(False if test_api.query_job() == "success" else True)])
    job.add_row(['job view', judging_state(test_api.job_api('view'))])
    job.add_row(['job list', judging_state(test_api.job_api('list/job'))])
    job.add_row(['job config', judging_state(test_api.job_config(max_iter=max_iter))])
    job.add_row(['job log', judging_state(test_api.job_api('log'))])
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
        ['query tag', judging_state(not test_api.tag_api('tag/query', 'update_job_tag') == 'update_job_tag')])
    tag.add_row(['delete tag', judging_state(test_api.tag_api('tag/delete', 'update_job_tag'))])
    print(tag.get_string(title="tag api"))

    component = PrettyTable()
    component.set_style(ORGMODE)
    component.field_names = ['component api name', 'status']
    component.add_row(['output data', judging_state(test_api.component_api('output/data'))])
    component.add_row(['output table', judging_state(test_api.component_api('output/data/table'))])
    component.add_row(['output model', judging_state(test_api.component_api('output/model'))])
    component.add_row(['component parameters', judging_state(test_api.component_api('parameters', max_iter=max_iter))])
    component.add_row(['component summary', judging_state(test_api.component_api('summary'))])
    component.add_row(['component list', judging_state(test_api.component_list())])
    component.add_row(['metrics', judging_state(test_api.component_api('metrics'))])
    component.add_row(['metrics all', judging_state(test_api.component_api('metric/all'))])

    model = PrettyTable()
    model.set_style(ORGMODE)
    model.field_names = ['model api name', 'status']
    model.add_row(['model load', judging_state(test_api.model_api('model/load'))])
    model.add_row(['model bind', judging_state(test_api.model_api('model/bind'))])
    status, model_path = test_api.model_api('model/export')
    model.add_row(['model export', judging_state(status)])
    model.add_row(['model  import', (judging_state(
        test_api.model_api('model/import', remove_path=remove_path, model_path=model_path)))])
    model.add_row(['tag model', judging_state(test_api.model_api('model_tag/model', tag_name='model_tag_create'))])
    model.add_row(['tag list', judging_state(test_api.model_api('model_tag/list', tag_name='model_tag_create'))])
    test_api.model_api('model_tag/model', tag_name='model_tag_create', remove=True)
    model.add_row(
        ['model migrate', judging_state(test_api.model_api('model/migrate'))])
    model.add_row(['model query', judging_state(test_api.model_api('model/query'))])
    model.add_row(['model deploy', judging_state(test_api.model_api('model/deploy'))])
    model.add_row(['model conf', judging_state(test_api.model_api('model/conf'))])
    model.add_row(['model dsl', judging_state(test_api.model_api('model/dsl'))])
    print(model.get_string(title="model api"))
    component.add_row(['metrics delete', judging_state(test_api.component_api('metric/delete'))])
    print(component.get_string(title="component api"))

    queue = PrettyTable()
    queue.set_style(ORGMODE)
    queue.field_names = ['api name', 'status']
    test_api.submit_job()
    test_api.submit_job()
    test_api.submit_job()
    queue.add_row(['clean/queue', judging_state(test_api.job_api('clean/queue'))])
    print(queue.get_string(title="queue job"))
