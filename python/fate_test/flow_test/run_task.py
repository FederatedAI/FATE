import json
import os
import time
from datetime import datetime

import requests
from fate_arch.common import conf_utils
from fate_flow.entity.types import StatusSet

config_path = 'config/settings.json'

ip = conf_utils.get_base_config("fateflow").get("host")
http_port = conf_utils.get_base_config("fateflow").get("http_port")
API_VERSION = "v1"
server_url = "http://{}:{}/{}".format(ip, http_port, API_VERSION)


def get_dict_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        json_info = json.load(f)
    return json_info


class Base(object):
    def __init__(self):
        self.config = None
        self.dsl = None
        self.guest_party_id = None
        self.host_party_id = None
        self.job_id = None
        self.model_id = None
        self.model_version = None

    def set_config(self, guest_party_id, host_party_id, path):
        self.config = get_dict_from_file(path)
        self.config["initiator"]["party_id"] = guest_party_id
        self.config["role"]["guest"] = [guest_party_id]
        self.config["role"]["host"] = [host_party_id]
        self.guest_party_id = guest_party_id
        self.host_party_id = host_party_id
        return self.config

    def set_dsl(self, path):
        self.dsl = get_dict_from_file(path)
        return self.dsl

    def submit(self, job_type='train'):
        post_data = {'job_runtime_conf': self.config, 'job_dsl': self.dsl}
        if job_type == 'predict':
            post_data.pop('job_dsl')
        print(f"start submit job, data:{post_data}")
        response = requests.post("/".join([server_url, "job", "submit"]), json=post_data)
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
        response = requests.post("/".join([server_url, "job", "query"]), json=post_data)
        if response.status_code == 200:
            if response.json().get("data"):
                return response.json().get("data")[0].get("f_status")
        return False

    def wait_success(self, timeout=60*10):
        for i in range(timeout//10):
            time.sleep(10)
            status = self.query_job()
            print("job {} status is {}".format(self.job_id, status))
            if status and status == StatusSet.SUCCESS:
                return True
            if status and status in [StatusSet.CANCELED, StatusSet.TIMEOUT, StatusSet.FAILED]:
                return False
        return False


class TrainSBTModel(Base):
    def get_component_metrics(self, file=None):
        post_data = {
            "job_id": self.job_id,
            "role": "guest",
            "party_id": self.guest_party_id,
            "component_name": "hetero_secure_boost_0"
        }
        response = requests.post("/".join([server_url, "tracking", "component/metric/all"]), json=post_data)
        if response.status_code == 200:
            if response.json().get("data"):
                if not file:
                    file = './output/metric/{}_hetero_secure_boost_0.json'.format(self.job_id)
                os.makedirs(os.path.dirname(file), exist_ok=True)
                with open(file, 'w') as fp:
                    json.dump(response.json().get("data"), fp)
                print(f"save component metrics success, path is:{os.path.abspath(file)}")
            else:
                print(f"get component metrics:{response.json()}")
                return False

    def get_component_output_model(self, file=None):
        post_data = {
            "job_id": self.job_id,
            "role": "guest",
            "party_id": self.guest_party_id,
            "component_name": "hetero_secure_boost_0"
        }
        print(f"request component output model: {post_data}")
        response = requests.post("/".join([server_url, "tracking", "component/output/model"]), json=post_data)
        if response.status_code == 200:
            if response.json().get("data"):
                if not file:
                    file = './output/model/{}_hetero_secure_boost_0.json'.format(self.job_id)
                os.makedirs(os.path.dirname(file), exist_ok=True)
                with open(file, 'w') as fp:
                    json.dump(response.json().get("data"), fp)
                print(f"save component output model success, path is:{os.path.abspath(file)}")
            else:
                print(f"get component output model:{response.json()}")
                return False


class PredictSBTMode(Base):
    def set_predict(self, guest_party_id, host_party_id, model_id, model_version, path):
        self.set_config(guest_party_id, host_party_id, path=path)
        self.config["job_parameters"]["common"]["model_id"] = model_id
        self.config["job_parameters"]["common"]["model_version"] = model_version


def train_job(guest_party_id, host_party_id, train_conf_path, train_dsl_path):
    train = TrainSBTModel()
    train.set_config(guest_party_id, host_party_id, train_conf_path)
    train.set_dsl(train_dsl_path)
    status = train.submit()
    if status:
        is_success = train.wait_success(timeout=600)
        if is_success:
            train.get_component_metrics()
            train.get_component_output_model()
            return train
    return False


def predict_job(guest_party_id, host_party_id, predict_conf_path, predict_dsl_path, model_id, model_version):
    predict = PredictSBTMode()
    predict.set_predict(guest_party_id, host_party_id, model_id, model_version, predict_conf_path)
    predict.set_dsl(predict_dsl_path)
    status = predict.submit(job_type='predict')
    if status:
        is_success = predict.wait_success(timeout=600)
        if is_success:
            return predict
    return False


class UtilizeModel:
    def __init__(self, model_id, model_version):
        self.model_id = model_id
        self.model_version = model_version
        self.deployed_model_version = None
        self.service_id = None
        self.settings = get_dict_from_file(config_path)

    def deploy_model(self):
        post_data = {
            "model_id": self.model_id,
            "model_version": self.model_version
        }
        response = requests.post("/".join([server_url, "model", "deploy"]), json=post_data)
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
        response = requests.post("/".join([server_url, "model", "load"]), json=post_data)
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
        response = requests.post("/".join([server_url, "model", "bind"]), json=post_data)
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

    def online_predict(self):
        serving_url = f"http://{self.settings.get('serving').get('host')}:{int(self.settings.get('serving').get('port'))}/federation/1.0/inference"
        post_data = {
            "head": {
                "serviceId": self.service_id
            },
            "body": {
                "featureData": {
                    "phone_num": "18576635456",
                },
                "sendToRemoteFeatureData": {
                  "device_type": "imei",
                  "phone_num": "18576635456",
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


def run_fate_flow_test():
    # train
    settings = get_dict_from_file(config_path)
    guest_party_id = settings.get("guest_party_id")
    host_party_id = settings.get("host_party_id")
    train_conf_path = settings.get("train_conf_path")
    train_dsl_path = settings.get("train_dsl_path")
    print('submit train job')
    train = train_job(guest_party_id, host_party_id, train_conf_path, train_dsl_path)
    if not train:
        print('train job run failed')
        return False
    print('train job success')

    # deploy
    print('start deploy model')
    utilize = UtilizeModel(train.model_id, train.model_version)
    utilize.deploy_model()
    print('deploy model success')

    # predict
    predict_conf_path = settings.get("predict_conf_path")
    predict_dsl_path = settings.get("predict_dsl_path")
    model_id = train.model_id
    model_version = utilize.deployed_model_version
    print('start submit predict job')
    predict = predict_job(guest_party_id, host_party_id, predict_conf_path, predict_dsl_path, model_id, model_version)
    if not predict:
        print('predict job run failed')
        return False
    print('predict job success')

    # load model
    utilize.load_model()

    # bind model
    utilize.bind_model()

    # online predict
    utilize.online_predict()


if __name__ == "__main__":
    run_fate_flow_test()









