import json
import os
import tarfile
import time
import subprocess
from contextlib import closing
from datetime import datetime

import requests


def get_dict_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        json_info = json.load(f)
    return json_info


def serving_connect(serving_setting):
    subp = subprocess.Popen([f'echo "" | telnet {serving_setting.split(":")[0]} {serving_setting.split(":")[1]}'],
                            shell=True, stdout=subprocess.PIPE)
    stdout, stderr = subp.communicate()
    stdout = stdout.decode("utf-8")
    return True if f'Connected to {serving_setting.split(":")[0]}' in stdout else False


class Base(object):
    def __init__(self, data_base_dir, server_url, component_name):
        self.config = None
        self.dsl = None
        self.guest_party_id = None
        self.host_party_id = None
        self.job_id = None
        self.model_id = None
        self.model_version = None

        self.data_base_dir = data_base_dir
        self.server_url = server_url
        self.component_name = component_name

    def set_config(self, guest_party_id, host_party_id, arbiter_party_id, path):
        self.config = get_dict_from_file(path)
        self.config["initiator"]["party_id"] = guest_party_id[0]
        self.config["role"]["guest"] = guest_party_id
        self.config["role"]["host"] = host_party_id
        if "arbiter" in self.config["role"]:
            self.config["role"]["arbiter"] = arbiter_party_id
        self.guest_party_id = guest_party_id
        self.host_party_id = host_party_id
        return self.config

    def set_dsl(self, path):
        self.dsl = get_dict_from_file(path)
        return self.dsl

    def submit(self):
        post_data = {'job_runtime_conf': self.config, 'job_dsl': self.dsl}
        print(f"start submit job, data:{post_data}")
        response = requests.post("/".join([self.server_url, "job", "submit"]), json=post_data)
        if response.status_code == 200 and not response.json().get('retcode'):
            self.job_id = response.json().get("jobId")
            print(f"submit job success: {response.json()}")
            self.model_id = response.json().get("data").get("model_info").get("model_id")
            self.model_version = response.json().get("data").get("model_info").get("model_version")
            return True
        else:
            print(f"submit job failed: {response.text}")
            return False

    def query_job(self):
        post_data = {'job_id': self.job_id}
        response = requests.post("/".join([self.server_url, "job", "query"]), json=post_data)
        if response.status_code == 200:
            if response.json().get("data"):
                return response.json().get("data")[0].get("f_status")
        return False

    def wait_success(self, timeout=60 * 10):
        for i in range(timeout // 10):
            time.sleep(10)
            status = self.query_job()
            print("job {} status is {}".format(self.job_id, status))
            if status and status == "success":
                return True
            if status and status in ["canceled", "timeout", "failed"]:
                return False
        return False

    def get_component_output_data(self, output_path=None):
        post_data = {
            "job_id": self.job_id,
            "role": "guest",
            "party_id": self.guest_party_id[0],
            "component_name": self.component_name
        }
        if not output_path:
            output_path = './output/data'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tar_file_name = 'job_{}_{}_{}_{}_output_data.tar.gz'.format(post_data['job_id'], post_data['component_name'],
                                                                    post_data['role'], post_data['party_id'])
        extract_dir = os.path.join(output_path, tar_file_name.replace('.tar.gz', ''))
        print("start get component output dat")

        with closing(
                requests.get("/".join([self.server_url, "tracking", "component/output/data/download"]), json=post_data,
                             stream=True)) as response:
            if response.status_code == 200:
                try:
                    download_from_request(http_response=response, tar_file_name=tar_file_name, extract_dir=extract_dir)
                    print(f'get component output path {extract_dir}')
                except BaseException:
                    print(f"get component output data failed")
                    return False

    def get_output_data_table(self):
        post_data = {
            "job_id": self.job_id,
            "role": "guest",
            "party_id": self.guest_party_id[0],
            "component_name": self.component_name
        }
        response = requests.post("/".join([self.server_url, "tracking", "component/output/data/table"]), json=post_data)
        result = {}
        try:
            if response.status_code == 200:
                result["name"] = response.json().get("data")[0].get("table_name")
                result["namespace"] = response.json().get("data")[0].get("namespace")
        except Exception as e:
            raise RuntimeError(f"output data table error: {response}") from e
        return result

    def get_table_info(self, table_name):
        post_data = {
            "name": table_name['name'],
            "namespace": table_name['namespace']
        }
        response = requests.post("/".join([self.server_url, "table", "table_info"]), json=post_data)
        try:
            if response.status_code == 200:
                table_count = response.json().get("data").get("count")
            else:
                raise RuntimeError(f"get table info failed: {response}")
        except Exception as e:
            raise RuntimeError(f"get table count error: {response}") from e
        return table_count

    def get_auc(self):
        post_data = {
            "job_id": self.job_id,
            "role": "guest",
            "party_id": self.guest_party_id[0],
            "component_name": "evaluation_0"
        }
        response = requests.post("/".join([self.server_url, "tracking", "component/metric/all"]), json=post_data)
        try:
            if response.status_code == 200:
                auc = response.json().get("data").get("train").get(self.component_name).get("data")[0][1]
            else:
                raise RuntimeError(f"get metrics failed: {response}")
        except Exception as e:
            raise RuntimeError(f"get table count error: {response}") from e
        return auc


class TrainLRModel(Base):
    def get_component_metrics(self, metric_output_path, file=None):
        post_data = {
            "job_id": self.job_id,
            "role": "guest",
            "party_id": self.guest_party_id[0],
            "component_name": "evaluation_0"
        }
        response = requests.post("/".join([self.server_url, "tracking", "component/metric/all"]), json=post_data)
        if response.status_code == 200:
            if response.json().get("data"):
                if not file:
                    file = metric_output_path.format(self.job_id)
                os.makedirs(os.path.dirname(file), exist_ok=True)
                with open(file, 'w') as fp:
                    json.dump(response.json().get("data"), fp)
                print(f"save component metrics success, path is:{os.path.abspath(file)}")
            else:
                print(f"get component metrics:{response.json()}")
                return False

    def get_component_output_model(self, model_output_path, file=None):
        post_data = {
            "job_id": self.job_id,
            "role": "guest",
            "party_id": self.guest_party_id[0],
            "component_name": self.component_name
        }
        print(f"request component output model: {post_data}")
        response = requests.post("/".join([self.server_url, "tracking", "component/output/model"]), json=post_data)
        if response.status_code == 200:
            if response.json().get("data"):
                if not file:
                    file = model_output_path.format(self.job_id)
                os.makedirs(os.path.dirname(file), exist_ok=True)
                with open(file, 'w') as fp:
                    json.dump(response.json().get("data"), fp)
                print(f"save component output model success, path is:{os.path.abspath(file)}")
            else:
                print(f"get component output model:{response.json()}")
                return False


class PredictLRMode(Base):
    def set_predict(self, guest_party_id, host_party_id, arbiter_party_id, model_id, model_version, path):
        self.set_config(guest_party_id, host_party_id, arbiter_party_id, path)
        if self.config["job_parameters"].get("common"):
            self.config["job_parameters"]["common"]["model_id"] = model_id
            self.config["job_parameters"]["common"]["model_version"] = model_version
        else:
            self.config["job_parameters"]["model_id"] = model_id
            self.config["job_parameters"]["model_version"] = model_version


def download_from_request(http_response, tar_file_name, extract_dir):
    with open(tar_file_name, 'wb') as fw:
        for chunk in http_response.iter_content(1024):
            if chunk:
                fw.write(chunk)
    tar = tarfile.open(tar_file_name, "r:gz")
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name, extract_dir)
    tar.close()
    os.remove(tar_file_name)


def train_job(data_base_dir, guest_party_id, host_party_id, arbiter_party_id, train_conf_path, train_dsl_path,
              server_url, component_name, metric_output_path, model_output_path, constant_auc):
    train = TrainLRModel(data_base_dir, server_url, component_name)
    train.set_config(guest_party_id, host_party_id, arbiter_party_id, train_conf_path)
    train.set_dsl(train_dsl_path)
    status = train.submit()
    if status:
        is_success = train.wait_success(timeout=600)
        if is_success:
            train.get_component_metrics(metric_output_path)
            train.get_component_output_model(model_output_path)
            train.get_component_output_data()
            train_auc = train.get_auc()
            assert abs(constant_auc - train_auc) <= 1e-4, 'The training result is wrong, auc: {}'.format(train_auc)
            train_data_count = train.get_table_info(train.get_output_data_table())
            return train, train_data_count
    return False


def predict_job(data_base_dir, guest_party_id, host_party_id, arbiter_party_id, predict_conf_path, predict_dsl_path,
                model_id, model_version, server_url, component_name):
    predict = PredictLRMode(data_base_dir, server_url, component_name)
    predict.set_predict(guest_party_id, host_party_id, arbiter_party_id, model_id, model_version, predict_conf_path)
    predict.set_dsl(predict_dsl_path)
    status = predict.submit()
    if status:
        is_success = predict.wait_success(timeout=600)
        if is_success:
            predict.get_component_output_data()
            predict_data_count = predict.get_table_info(predict.get_output_data_table())
            return predict, predict_data_count
    return False


class UtilizeModel:
    def __init__(self, model_id, model_version, server_url):
        self.model_id = model_id
        self.model_version = model_version
        self.deployed_model_version = None
        self.service_id = None
        self.server_url = server_url

    def deploy_model(self):
        post_data = {
            "model_id": self.model_id,
            "model_version": self.model_version
        }
        response = requests.post("/".join([self.server_url, "model", "deploy"]), json=post_data)
        print(f'Request data of deploy model request: {json.dumps(post_data, indent=4)}')
        if response.status_code == 200:
            resp_data = response.json()
            print(f'Response of model deploy request: {json.dumps(resp_data, indent=4)}')
            if resp_data.get("retcode", 100) == 0:
                self.deployed_model_version = resp_data.get("data", {}).get("model_version")
            else:
                raise Exception(f"Model {self.model_id} {self.model_version} deploy failed, "
                                f"details: {resp_data.get('retmsg')}")
        else:
            raise Exception(f"Request model deploy api failed, status code: {response.status_code}")

    def load_model(self):
        post_data = {
            "job_id": self.deployed_model_version
        }
        response = requests.post("/".join([self.server_url, "model", "load"]), json=post_data)
        print(f'Request data of load model request: {json.dumps(post_data, indent=4)}')
        if response.status_code == 200:
            resp_data = response.json()
            print(f'Response of load model request: {json.dumps(resp_data, indent=4)}')
            if not resp_data.get('retcode'):
                return True
            raise Exception(f"Load model {self.model_id} {self.deployed_model_version} failed, "
                            f"details: {resp_data.get('retmsg')}")
        raise Exception(f"Request model load api failed, status code: {response.status_code}")

    def bind_model(self):
        post_data = {
            "job_id": self.deployed_model_version,
            "service_id": f"auto_test_{datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')}"
        }
        response = requests.post("/".join([self.server_url, "model", "bind"]), json=post_data)
        print(f'Request data of bind model request: {json.dumps(post_data, indent=4)}')
        if response.status_code == 200:
            resp_data = response.json()
            print(f'Response data of bind model request: {json.dumps(resp_data, indent=4)}')
            if not resp_data.get('retcode'):
                self.service_id = post_data.get('service_id')
                return True
            raise Exception(f"Bind model {self.model_id} {self.deployed_model_version} failed, "
                            f"details: {resp_data.get('retmsg')}")
        raise Exception(f"Request model bind api failed, status code: {response.status_code}")

    def online_predict(self, online_serving, phone_num):
        serving_url = f"http://{online_serving}/federation/1.0/inference"
        post_data = {
            "head": {
                "serviceId": self.service_id
            },
            "body": {
                "featureData": {
                    "phone_num": phone_num,
                },
                "sendToRemoteFeatureData": {
                    "device_type": "imei",
                    "phone_num": phone_num,
                    "encrypt_type": "raw"
                }
            }
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(serving_url, json=post_data, headers=headers)
        print(f"Request data of online predict request: {json.dumps(post_data, indent=4)}")
        if response.status_code == 200:
            print(f"Online predict successfully, response: {json.dumps(response.json(), indent=4)}")
        else:
            print(f"Online predict successfully, details: {response.text}")


def run_fate_flow_test(config_json):
    data_base_dir = config_json['data_base_dir']
    guest_party_id = config_json['guest_party_id']
    host_party_id = config_json['host_party_id']
    arbiter_party_id = config_json['arbiter_party_id']
    train_conf_path = config_json['train_conf_path']
    train_dsl_path = config_json['train_dsl_path']
    server_url = config_json['server_url']
    online_serving = config_json['online_serving']
    constant_auc = config_json['train_auc']
    component_name = config_json['component_name']
    metric_output_path = config_json['metric_output_path']
    model_output_path = config_json['model_output_path']
    serving_connect_bool = serving_connect(config_json['serving_setting'])
    phone_num = config_json['phone_num']

    print('submit train job')
    # train
    train, train_count = train_job(data_base_dir, guest_party_id, host_party_id, arbiter_party_id, train_conf_path,
                                   train_dsl_path, server_url, component_name, metric_output_path, model_output_path, constant_auc)
    if not train:
        print('train job run failed')
        return False
    print('train job success')

    # deploy
    print('start deploy model')
    utilize = UtilizeModel(train.model_id, train.model_version, server_url)
    utilize.deploy_model()
    print('deploy model success')

    # predict
    predict_conf_path = config_json['predict_conf_path']
    predict_dsl_path = config_json['predict_dsl_path']
    model_id = train.model_id
    model_version = utilize.deployed_model_version
    print('start submit predict job')
    predict, predict_count = predict_job(data_base_dir, guest_party_id, host_party_id, arbiter_party_id, predict_conf_path,
                                         predict_dsl_path, model_id, model_version, server_url, component_name)
    if not predict:
        print('predict job run failed')
        return False
    if train_count != predict_count:
        print('Loss of forecast data')
        return False
    print('predict job success')

    if not config_json.get('component_is_homo') and serving_connect_bool:
        # load model
        utilize.load_model()

        # bind model
        utilize.bind_model()

        # online predict
        utilize.online_predict(online_serving=online_serving, phone_num=phone_num)
