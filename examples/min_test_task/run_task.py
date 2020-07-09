import argparse
import json
import os
import random
import subprocess
import time

home_dir = os.path.split(os.path.realpath(__file__))[0]

# Hetero-lr task
hetero_lr_config_file = home_dir + "/config/test_hetero_lr_train_job_conf.json"
hetero_lr_dsl_file = home_dir + "/config/test_hetero_lr_train_job_dsl.json"

# hetero-sbt task
hetero_sbt_config_file = home_dir + "/config/test_secureboost_train_binary_conf.json"
hetero_sbt_dsl_file = home_dir + "/config/test_secureboost_train_dsl.json"

predict_task_file = home_dir + "/config/test_predict_conf.json"

guest_import_data_file = home_dir + "/config/data/breast_b.csv"
fate_flow_path = home_dir + "/../../fate_flow/fate_flow_client.py"

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


def get_timeid():
    return str(int(time.time())) + "_" + str(random.randint(1000, 9999))


def gen_unique_path(prefix):
    return home_dir + "/test/" + prefix + ".config_" + get_timeid()


def time_print(msg):
    print(f"[{time.strftime('%Y-%m-%d %X')}] {msg}\n")


class TaskManager(object):
    @staticmethod
    def start_block_task(cmd, max_waiting_time=OTHER_TASK_TIME):
        start_time = time.time()
        print(f"Starting block task, cmd is {cmd}")
        while True:
            # print("exec cmd: {}".format(cmd))
            subp = subprocess.Popen(cmd,
                                    shell=False,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            stdout, stderr = subp.communicate()
            stdout = stdout.decode("utf-8")
            if not stdout:
                waited_time = time.time() - start_time
                if waited_time >= max_waiting_time:
                    # raise ValueError(
                    #     "[obtain_component_output] task:{} failed stdout:{}".format(task_type, stdout))
                    return None
                print("job cmd: {}, waited time: {}".format(cmd, waited_time))
                time.sleep(STATUS_CHECKER_TIME)
            else:
                break
        try:
            stdout = json.loads(stdout)
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
    def start_task(cmd):
        time_print('Start task: {}'.format(cmd))
        subp = subprocess.Popen(cmd,
                                shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        stdout, stderr = subp.communicate()
        stdout = stdout.decode("utf-8")
        # time_print("start_task, stdout:" + str(stdout))
        try:
            stdout = json.loads(stdout)
        except:
            raise RuntimeError("start task error, return value: {}".format(stdout))
        return stdout

    def get_table_info(self, name, namespace):
        cmd = ["python", fate_flow_path, "-f", "table_info", "-t", str(name), "-n", str(namespace)]
        table_info = self.start_task(cmd)
        time_print(table_info)
        return table_info


class TrainTask(TaskManager):
    def __init__(self, data_type, guest_id, host_id, arbiter_id, work_mode):
        self.method = 'all'
        self.guest_id = guest_id
        self.host_id = host_id
        self.arbiter_id = arbiter_id
        self.work_mode = work_mode
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

    def _check_status(self, jobid):
        pass

    def run(self):
        config_dir_path = self._make_runtime_conf()
        start_task_cmd = ['python', fate_flow_path, "-f", "submit_job", "-c",
                          config_dir_path, "-d", self.dsl_file]
        stdout = self.start_task(start_task_cmd)
        status = stdout["retcode"]

        if status != 0:
            raise ValueError(
                "Training task exec fail, status:{}, stdout:{}".format(status, stdout))
        else:
            jobid = stdout["jobId"]

        self.model_id = stdout['data']['model_info']['model_id']
        self.model_version = stdout['data']['model_info']['model_version']

        self._check_status(jobid)

        auc = self._get_auc(jobid)
        if auc < self.auc_base:
            time_print("[Warning]  The auc: {} is lower than expect value: {}".format(auc, self.auc_base))
        else:
            time_print("[Train] train auc:{}".format(auc))
        time.sleep(WAIT_UPLOAD_TIME / 100)
        self.start_predict_task()

    def start_predict_task(self):
        config_dir_path = self._make_runtime_conf("predict")
        start_task_cmd = ['python', fate_flow_path, "-f", "submit_job", "-c", config_dir_path]
        stdout = self.start_task(start_task_cmd)
        status = stdout["retcode"]
        if status != 0:
            raise ValueError(
                "Training task exec fail, status:{}, stdout:{}".format(status, stdout))
        else:
            jobid = stdout["jobId"]

        self._check_status(jobid)

        time_print("[Predict Task] Predict success")

    def _parse_dsl_components(self):
        with open(hetero_lr_dsl_file, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())
        components = list(json_info['components'].keys())
        return components

    def _check_cpn_status(self, job_id):
        check_cmd = ['python', fate_flow_path, "-f", "query_job",
                     "-j", job_id, "-r", "guest"]

        stdout = self.start_task(check_cmd)
        try:
            status = stdout["retcode"]
            if status != 0:
                return RUNNING
            # time_print("In _check_cpn_status, status: {}".format(status))
            check_data = stdout["data"]
            task_status = check_data[0]['f_status']

            time_print("Current task status: {}".format(task_status))
            return task_status
        except:
            return None

    @staticmethod
    def _check_exit(status):
        if status is None:
            return True

        if status in [RUNNING, START, WAITING]:
            return False
        return True

    def _get_auc(self, jobid):
        cmd = ["python", fate_flow_path, "-f", "component_metric_all", "-j",
               jobid, "-p", str(self.guest_id), "-r", "guest",
               "-cpn", evaluation_component_name]
        eval_res = self.start_block_task(cmd, max_waiting_time=OTHER_TASK_TIME)
        eval_results = eval_res['data']['train'][self.train_component_name]['data']
        time_print("Get auc eval res: {}".format(eval_results))
        auc = 0
        for metric_name, metric_value in eval_results:
            if metric_name == 'auc':
                auc = metric_value
        return auc


class TrainLRTask(TrainTask):
    def __init__(self, data_type, guest_id, host_id, arbiter_id, work_mode):
        super().__init__(data_type, guest_id, host_id, arbiter_id, work_mode)
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
        json_info['job_parameters']['work_mode'] = self.work_mode

        if self.model_id is not None:
            json_info["job_parameters"]["model_id"] = self.model_id
            json_info["job_parameters"]["model_version"] = self.model_version

        table_info = {"name": self.guest_table_name,
                      "namespace": self.guest_namespace}
        if conf_type == 'train':
            json_info["role_parameters"]["guest"]["args"]["data"]["train_data"] = [table_info]
            json_info["role_parameters"]["guest"]["args"]["data"]["eval_data"] = [table_info]
        else:
            json_info["role_parameters"]["guest"]["args"]["data"]["eval_data"] = [table_info]

        table_info = {"name": self.host_name,
                      "namespace": self.host_namespace}
        if conf_type == 'train':
            json_info["role_parameters"]["host"]["args"]["data"]["train_data"] = [table_info]
            json_info["role_parameters"]["host"]["args"]["data"]["eval_data"] = [table_info]
        else:
            json_info["role_parameters"]["host"]["args"]["data"]["eval_data"] = [table_info]

        config = json.dumps(json_info)
        config_path = gen_unique_path('submit_job_guest')
        config_dir_path = os.path.dirname(config_path)
        os.makedirs(config_dir_path, exist_ok=True)
        with open(config_path, "w") as fout:
            fout.write(config + "\n")
        return config_path

    def _check_status(self, jobid):
        params = [jobid]
        job_status = self.start_block_func(self._check_cpn_status, params,
                                           exit_func=self._check_exit, max_waiting_time=MAX_TRAIN_TIME)
        if job_status == FAIL:
            exit(1)


class TrainSBTTask(TrainTask):
    def __init__(self, data_type, guest_id, host_id, arbiter_id, work_mode):
        super().__init__(data_type, guest_id, host_id, arbiter_id, work_mode)
        self.dsl_file = hetero_sbt_dsl_file
        self.train_component_name = 'secureboost_0'

    def _make_runtime_conf(self, conf_type='train'):
        if conf_type == 'train':
            input_template = hetero_sbt_config_file
        else:
            input_template = predict_task_file
        with open(input_template, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())

        json_info['role']['guest'] = [self.guest_id]
        json_info['role']['host'] = [self.host_id]

        json_info['initiator']['party_id'] = self.guest_id
        json_info['job_parameters']['work_mode'] = self.work_mode

        if self.model_id is not None:
            json_info["job_parameters"]["model_id"] = self.model_id
            json_info["job_parameters"]["model_version"] = self.model_version

        table_info = {"name": self.guest_table_name,
                      "namespace": self.guest_namespace}
        if conf_type == 'train':
            json_info["role_parameters"]["guest"]["args"]["data"]["train_data"] = [table_info]
            json_info["role_parameters"]["guest"]["args"]["data"]["eval_data"] = [table_info]
        else:
            json_info["role_parameters"]["guest"]["args"]["data"]["eval_data"] = [table_info]

        table_info = {"name": self.host_name,
                      "namespace": self.host_namespace}
        if conf_type == 'train':
            json_info["role_parameters"]["host"]["args"]["data"]["train_data"] = [table_info]
            json_info["role_parameters"]["host"]["args"]["data"]["eval_data"] = [table_info]
        else:
            json_info["role_parameters"]["host"]["args"]["data"]["eval_data"] = [table_info]

        config = json.dumps(json_info)
        config_path = gen_unique_path('submit_job_guest')
        config_dir_path = os.path.dirname(config_path)
        os.makedirs(config_dir_path, exist_ok=True)
        with open(config_path, "w") as fout:
            fout.write(config + "\n")
        return config_path

    def _check_status(self, jobid):
        params = [jobid]
        job_status = self.start_block_func(self._check_cpn_status, params,
                                           exit_func=self._check_exit, max_waiting_time=MAX_TRAIN_TIME)
        if job_status == FAIL:
            exit(1)


def main():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-m", "--mode", type=int, help="work mode", choices=[0, 1], required=True)
    arg_parser.add_argument("-f", "--file_type", type=str,
                            help="file_type, "
                                 "'fast' means breast data "
                                 "'normal' means default credit data",
                            choices=["fast", "normal"],
                            default="fast")

    arg_parser.add_argument("-gid", "--guest_id", type=int, help="guest party id", required=True)
    arg_parser.add_argument("-hid", "--host_id", type=int, help="host party id", required=True)
    arg_parser.add_argument("-aid", "--arbiter_id", type=int, help="arbiter party id", required=True)

    arg_parser.add_argument("--add_sbt", help="test sbt or not", type=int,
                            default=1, choices=[0, 1])

    args = arg_parser.parse_args()

    work_mode = args.mode
    guest_id = args.guest_id
    host_id = args.host_id
    arbiter_id = args.arbiter_id
    file_type = args.file_type
    add_sbt = args.add_sbt

    task = TrainLRTask(file_type, guest_id, host_id, arbiter_id, work_mode)
    task.run()

    if add_sbt:
        task = TrainSBTTask(file_type, guest_id, host_id, arbiter_id, work_mode)
        task.run()


if __name__ == "__main__":
    main()
