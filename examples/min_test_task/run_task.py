import argparse
import json
import os
import random
import time

from flow_sdk.client import FlowClient

home_dir = os.path.split(os.path.realpath(__file__))[0]

# Hetero-lr task
hetero_lr_config_file = home_dir + "/config/test_hetero_lr_train_job_conf.json"
hetero_lr_dsl_file = home_dir + "/config/test_hetero_lr_train_job_dsl.json"

# hetero-sbt task
hetero_sbt_config_file = home_dir + "/config/test_secureboost_train_binary_conf.json"
hetero_sbt_dsl_file = home_dir + "/config/test_secureboost_train_dsl.json"

publish_conf_file = home_dir + "/config/publish_load_model.json"
bind_conf_file = home_dir + "/config/bind_model_service.json"

predict_task_file = home_dir + "/config/test_predict_conf.json"

guest_import_data_file = home_dir + "/config/data/breast_b.csv"
# fate_flow_path = home_dir + "/../../python/fate_flow/fate_flow_client.py"
fate_flow_home = home_dir + "/../../python/fate_flow"

evaluation_component_name = 'evaluation_0'
# GUEST = 'guest'
# HOST = 'host'
# ARBITER = 'arbiter'

START = 'start'
SUCCESS = 'success'
RUNNING = 'running'
WAITING = 'waiting'
FAIL = 'failed'
STUCK = 'stuck'
# READY = 'ready'
MAX_INTERSECT_TIME = 3600
MAX_TRAIN_TIME = 7200
WAIT_UPLOAD_TIME = 1000
OTHER_TASK_TIME = 7200
# RETRY_JOB_STATUS_TIME = 5
STATUS_CHECKER_TIME = 10
flow_client: FlowClient


def get_flow_info():
    from fate_flow import set_env
    from fate_arch.common.conf_utils import get_base_config
    FATE_FLOW_SERVICE_NAME = "fateflow"
    HOST = get_base_config(FATE_FLOW_SERVICE_NAME, {}).get("host", "127.0.0.1")
    HTTP_PORT = get_base_config(FATE_FLOW_SERVICE_NAME, {}).get("http_port")
    return HOST, HTTP_PORT


def get_timeid():
    return str(int(time.time())) + "_" + str(random.randint(1000, 9999))


def gen_unique_path(prefix):
    return home_dir + "/test/" + prefix + ".config_" + get_timeid()


def time_print(msg):
    print(f"[{time.strftime('%Y-%m-%d %X')}] {msg}\n")


def get_config_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        json_info = json.load(f)
    return json_info


class TaskManager(object):
    @staticmethod
    def start_block_task(cmd, max_waiting_time=OTHER_TASK_TIME):
        start_time = time.time()
        print(f"Starting block task, cmd is {cmd}")
        while True:
            # print("exec cmd: {}".format(cmd))
            stdout = flow_client.component.metric_all(job_id=cmd.get("job_id"), role="guest",
                                                      party_id=cmd.get("party_id"),
                                                      component_name=cmd.get("component_name"))

            if not stdout:
                waited_time = time.time() - start_time
                if waited_time >= max_waiting_time:
                    # raise ValueError(
                    #     "[obtain_component_output] task:{} failed stdout:{}".format(task_type, stdout))
                    return None
                print("job cmd: component metric_all, waited time: {}".format(waited_time))
                time.sleep(STATUS_CHECKER_TIME)
            else:
                break
        try:
            json.dumps(stdout)
        except json.decoder.JSONDecodeError:
            raise RuntimeError("start task error, return value: {}".format(stdout))

        return stdout

    @staticmethod
    def start_block_func(run_func, params, exit_func, max_waiting_time=OTHER_TASK_TIME):
        start_time = time.time()
        while True:
            result = run_func(*params)
            if exit_func(result):
                return result
            end_time = time.time()
            if end_time - start_time >= max_waiting_time:
                return None
            time.sleep(STATUS_CHECKER_TIME)

    @staticmethod
    def task_status(stdout, msg):
        status = stdout.get("retcode", None)
        if status is None:
            raise RuntimeError("start task error, return value: {}".format(stdout))
        elif status == 0:
            return status
        else:
            raise ValueError("{}, status:{}, stdout:{}".format(msg, status, stdout))

    def get_table_info(self, name, namespace):
        time_print('Start task: {}'.format("table_info"))
        stdout = flow_client.table.info(namespace=str(namespace), table_name=str(name))
        self.task_status(stdout, "query data info task exec fail")

        return stdout


class TrainTask(TaskManager):
    def __init__(self, data_type, guest_id, host_id, arbiter_id=0):
        self.method = 'all'
        self.guest_id = guest_id
        self.host_id = host_id
        self.arbiter_id = arbiter_id
        self._data_type = data_type
        self.model_id = None
        self.model_version = None
        self.dsl_file = None
        self.train_component_name = None
        self._parse_argv()

    def _parse_argv(self):

        if self._data_type == 'fast':
            self.task_data_count = 569
            self.task_intersect_count = 569
            self.auc_base = 0.98
            self.guest_table_name = "breast_hetero_guest"
            self.guest_namespace = "experiment"
            self.host_name = "breast_hetero_host"
            self.host_namespace = "experiment"
        elif self._data_type == "normal":
            self.task_data_count = 30000
            self.task_intersect_count = 30000
            self.auc_base = 0.69
            self.guest_table_name = "default_credit_hetero_guest"
            self.guest_namespace = "experiment"
            self.host_name = "default_credit_hetero_host"
            self.host_namespace = "experiment"
        else:
            raise ValueError("Unknown data type:{}".format(self._data_type))

    def _make_runtime_conf(self, conf_type='train'):
        pass

    def _check_status(self, job_id):
        pass

    # def _deploy_model(self):
    #     pass

    def run(self, start_serving=0):
        config_dir_path = self._make_runtime_conf()
        time_print('Start task: {}'.format("job submit"))
        stdout = flow_client.job.submit(config_data=get_config_file(config_dir_path),
                                        dsl_data=get_config_file(self.dsl_file))
        self.task_status(stdout, "Training task exec fail")
        print(json.dumps(stdout, indent=4))
        job_id = stdout.get("jobId")

        self.model_id = stdout['data']['model_info']['model_id']
        self.model_version = stdout['data']['model_info']['model_version']

        self._check_status(job_id)
        auc = self._get_auc(job_id)
        if auc < self.auc_base:
            time_print("[Warning]  The auc: {} is lower than expect value: {}".format(auc, self.auc_base))
        else:
            time_print("[Train] train auc:{}".format(auc))
        time.sleep(WAIT_UPLOAD_TIME / 100)
        self.start_predict_task()

        if start_serving:
            self._load_model()
            self._bind_model()

    def start_predict_task(self):
        self._deploy_model()
        config_dir_path = self._make_runtime_conf("predict")
        time_print('Start task: {}'.format("job submit"))
        stdout = flow_client.job.submit(config_data=get_config_file(config_dir_path))
        self.task_status(stdout, "Training task exec fail")
        job_id = stdout.get("jobId")
        self._check_status(job_id)
        time_print("[Predict Task] Predict success")

    @staticmethod
    def _parse_dsl_components():
        with open(hetero_lr_dsl_file, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())
        components = list(json_info['components'].keys())
        return components

    @staticmethod
    def _check_cpn_status(job_id):
        time_print('Start task: {}'.format("job query"))
        stdout = flow_client.job.query(job_id=job_id)
        try:
            status = stdout["retcode"]
            if status != 0:
                return RUNNING
            # time_print("In _check_cpn_status, status: {}".format(status))
            check_data = stdout["data"]
            task_status = check_data[0]['f_status']

            time_print("Current task status: {}".format(task_status))
            return task_status
        except BaseException:
            return None

    def _deploy_model(self):
        time_print('Start task: {}'.format("model deploy"))
        stdout = flow_client.model.deploy(model_id=self.model_id, model_version=self.model_version, cpn_list=[
            "reader_0", "data_transform_0", "intersection_0", self.train_component_name])
        self.task_status(stdout, "Deploy task exec fail")
        time_print(stdout)
        self.predict_model_version = stdout["data"]["model_version"]
        self.predict_model_id = stdout["data"]["model_id"]
        time_print("[Predict Task] Deploy success")

    @staticmethod
    def _check_exit(status):
        if status is None:
            return True

        if status in [RUNNING, START, WAITING]:
            return False
        return True

    def _get_auc(self, job_id):
        cmd = {'job_id': job_id, "party_id": str(self.guest_id), "component_name": evaluation_component_name}
        eval_res = self.start_block_task(cmd, max_waiting_time=OTHER_TASK_TIME)
        eval_results = eval_res['data']['train'][self.train_component_name]['data']
        time_print("Get auc eval res: {}".format(eval_results))
        auc = 0
        for metric_name, metric_value in eval_results:
            if metric_name == 'auc':
                auc = metric_value
        return auc

    def _bind_model(self):
        config_path = self.__config_bind_load(bind_conf_file)
        time_print('Start task: {}'.format("model bind"))
        stdout = flow_client.model.bind(config_data=get_config_file(config_path))
        self.task_status(stdout, "Bind model failed")
        time_print("Bind model Success")
        return True

    def __config_bind_load(self, template):
        with open(template, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())
        json_info["service_id"] = self.model_id
        json_info["initiator"]["party_id"] = str(self.guest_id)
        json_info["role"]["guest"] = [str(self.guest_id)]
        json_info["role"]["host"] = [str(self.host_id)]
        json_info["role"]["arbiter"] = [str(self.arbiter_id)]
        json_info["job_parameters"]["model_id"] = self.predict_model_id
        json_info["job_parameters"]["model_version"] = self.predict_model_version

        if 'servings' in json_info:
            del json_info['servings']

        config = json.dumps(json_info, indent=4)
        config_path = gen_unique_path('bind_model')
        config_dir_path = os.path.dirname(config_path)
        os.makedirs(config_dir_path, exist_ok=True)
        with open(config_path, "w") as fout:
            fout.write(config + "\n")
        return config_path

    def _load_model(self):
        config_path = self.__config_bind_load(publish_conf_file)
        time_print('Start task: {}'.format("model load"))
        stdout = flow_client.model.load(config_data=get_config_file(config_path))
        status = self.task_status(stdout, "Load model failed")

        data = stdout["data"]
        try:
            guest_retcode = data["detail"]["guest"][str(self.guest_id)]["retcode"]
            host_retcode = data["detail"]["host"][str(self.host_id)]["retcode"]
        except KeyError:
            raise ValueError(
                "Load model failed, status:{}, stdout:{}".format(status, stdout))
        if guest_retcode != 0 or host_retcode != 0:
            raise ValueError(
                "Load model failed, status:{}, stdout:{}".format(status, stdout))
        time_print("Load model Success")

        return True


class TrainLRTask(TrainTask):
    def __init__(self, data_type, guest_id, host_id, arbiter_id):
        super().__init__(data_type, guest_id, host_id, arbiter_id)

        self.dsl_file = hetero_lr_dsl_file
        self.train_component_name = 'hetero_lr_0'

    def _make_runtime_conf(self, conf_type='train'):
        if conf_type == 'train':
            input_template = hetero_lr_config_file
        else:
            input_template = predict_task_file
        with open(input_template, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())

        json_info['role']['guest'] = [self.guest_id]
        json_info['role']['host'] = [self.host_id]
        json_info['role']['arbiter'] = [self.arbiter_id]

        json_info['initiator']['party_id'] = self.guest_id

        if self.model_id is not None:
            json_info["job_parameters"]["common"]["model_id"] = self.predict_model_id
            json_info["job_parameters"]["common"]["model_version"] = self.predict_model_version

        table_info = {"name": self.guest_table_name,
                      "namespace": self.guest_namespace}
        if conf_type == 'train':
            json_info["component_parameters"]["role"]["guest"]["0"]["reader_0"]["table"] = table_info
            json_info["component_parameters"]["role"]["guest"]["0"]["reader_1"]["table"] = table_info
        else:
            json_info["component_parameters"]["role"]["guest"]["0"]["reader_0"]["table"] = table_info

        table_info = {"name": self.host_name,
                      "namespace": self.host_namespace}
        if conf_type == 'train':
            json_info["component_parameters"]["role"]["host"]["0"]["reader_0"]["table"] = table_info
            json_info["component_parameters"]["role"]["host"]["0"]["reader_1"]["table"] = table_info
        else:
            json_info["component_parameters"]["role"]["host"]["0"]["reader_0"]["table"] = table_info

        config = json.dumps(json_info, indent=4)
        config_path = gen_unique_path('submit_job_guest')
        config_dir_path = os.path.dirname(config_path)
        os.makedirs(config_dir_path, exist_ok=True)
        with open(config_path, "w") as fout:
            fout.write(config + "\n")
        return config_path

    def _check_status(self, job_id):
        params = [job_id]
        job_status = self.start_block_func(self._check_cpn_status, params,
                                           exit_func=self._check_exit, max_waiting_time=MAX_TRAIN_TIME)
        if job_status == FAIL:
            exit(1)


class TrainSBTTask(TrainTask):
    def __init__(self, data_type, guest_id, host_id):
        super().__init__(data_type, guest_id, host_id)
        self.dsl_file = hetero_sbt_dsl_file
        self.train_component_name = 'hetero_secure_boost_0'

    def _make_runtime_conf(self, conf_type='train'):
        if conf_type == 'train':
            input_template = hetero_sbt_config_file
        else:
            input_template = predict_task_file
        with open(input_template, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())

        json_info['role']['guest'] = [self.guest_id]
        json_info['role']['host'] = [self.host_id]

        if 'arbiter' in json_info['role']:
            del json_info['role']['arbiter']

        json_info['initiator']['party_id'] = self.guest_id

        if self.model_id is not None:
            json_info["job_parameters"]["common"]["model_id"] = self.predict_model_id
            json_info["job_parameters"]["common"]["model_version"] = self.predict_model_version

        table_info = {"name": self.guest_table_name,
                      "namespace": self.guest_namespace}
        if conf_type == 'train':
            json_info["component_parameters"]["role"]["guest"]["0"]["reader_0"]["table"] = table_info
            json_info["component_parameters"]["role"]["guest"]["0"]["reader_1"]["table"] = table_info
        else:
            json_info["component_parameters"]["role"]["guest"]["0"]["reader_0"]["table"] = table_info

        table_info = {"name": self.host_name,
                      "namespace": self.host_namespace}
        if conf_type == 'train':
            json_info["component_parameters"]["role"]["host"]["0"]["reader_0"]["table"] = table_info
            json_info["component_parameters"]["role"]["host"]["0"]["reader_1"]["table"] = table_info
        else:
            json_info["component_parameters"]["role"]["host"]["0"]["reader_0"]["table"] = table_info

        config = json.dumps(json_info, indent=4)
        config_path = gen_unique_path('submit_job_guest')
        config_dir_path = os.path.dirname(config_path)
        os.makedirs(config_dir_path, exist_ok=True)
        with open(config_path, "w") as fout:
            fout.write(config + "\n")
        return config_path

    def _check_status(self, job_id):
        params = [job_id]
        job_status = self.start_block_func(self._check_cpn_status, params,
                                           exit_func=self._check_exit, max_waiting_time=MAX_TRAIN_TIME)
        if job_status == FAIL:
            exit(1)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-f", "--file_type", type=str,
                            help="file_type, "
                                 "'fast' means breast data "
                                 "'normal' means default credit data",
                            choices=["fast", "normal"],
                            default="fast")

    arg_parser.add_argument("-gid", "--guest_id", type=int, help="guest party id", required=True)
    arg_parser.add_argument("-hid", "--host_id", type=int, help="host party id", required=True)
    arg_parser.add_argument("-aid", "--arbiter_id", type=int, help="arbiter party id", required=True)
    arg_parser.add_argument("-ip", "--flow_server_ip", type=str, help="please input flow server'ip")
    arg_parser.add_argument("-port", "--flow_server_port", type=int, help="please input flow server port")
    arg_parser.add_argument("--add_sbt", help="test sbt or not", type=int,
                            default=1, choices=[0, 1])

    arg_parser.add_argument("-s", "--serving", type=int, help="Test Serving process",
                            default=0, choices=[0, 1])

    args = arg_parser.parse_args()

    guest_id = args.guest_id
    host_id = args.host_id
    arbiter_id = args.arbiter_id
    file_type = args.file_type
    add_sbt = args.add_sbt
    start_serving = args.serving
    ip = args.flow_server_ip
    port = args.flow_server_port
    if ip is None:
        ip, port = get_flow_info()
    flow_client = FlowClient(ip=ip, port=port, version="v1")
    task = TrainLRTask(file_type, guest_id, host_id, arbiter_id)
    task.run(start_serving)

    if add_sbt:
        task = TrainSBTTask(file_type, guest_id, host_id)
        task.run()
