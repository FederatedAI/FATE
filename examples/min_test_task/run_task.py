import json
import os
import random
import subprocess
import sys
import time

home_dir = os.path.split(os.path.realpath(__file__))[0]

# Upload and download data
upload_config_file = home_dir + "/config/upload.json"
download_config_file = home_dir + "/config/download.json"

# Hetero-lr task
hetero_lr_config_file = home_dir + "/config/test_hetero_lr_train_job_conf.json"
hetero_lr_dsl_file = home_dir + "/config/test_hetero_lr_train_job_dsl.json"

guest_import_data_file = home_dir + "/config/data/breast_b.csv"
fate_flow_path = home_dir + "/../../fate_flow/fate_flow_client.py"

guest_id = 9999
host_id = 10000
arbiter_id = 10000

work_mode = 1

intersect_output_name = ''
intersect_output_namespace = ''
eval_output_name = ''
eval_output_namespace = ''

train_component_name = 'hetero_lr_0'
evaluation_component_name = 'evaluation_0'

GUEST = 'guest'
HOST = 'host'
ARBITER = 'arbiter'

START = 'start'
SUCCESS = 'success'
RUNNING = 'running'
FAIL = 'failed'
STUCK = 'stuck'
# READY = 'ready'
MAX_INTERSECT_TIME = 600
MAX_TRAIN_TIME = 3600
OTHER_TASK_TIME = 300
# RETRY_JOB_STATUS_TIME = 5
STATUS_CHECKER_TIME = 10


def get_timeid():
    return str(int(time.time())) + "_" + str(random.randint(1000, 9999))


def gen_unique_path(prefix):
    return home_dir + "/test/" + prefix + ".config_" + get_timeid()


class TaskManager(object):
    def __init__(self, argv=None):
        if argv is not None:
            self._parse_argv(argv)

    def _parse_argv(self, argv):
        raise NotImplementedError("Should not call here")

    @staticmethod
    def start_block_task(cmd, max_waiting_time=OTHER_TASK_TIME):
        start_time = time.time()
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
        stdout = json.loads(stdout)
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
        print('Start task: {}'.format(cmd))
        subp = subprocess.Popen(cmd,
                                shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        stdout, stderr = subp.communicate()
        stdout = stdout.decode("utf-8")
        # print("start_task, stdout:" + str(stdout))
        try:
            stdout = json.loads(stdout)
        except:
            raise RuntimeError("start task error, return value: {}".format(stdout))
        return stdout

    def get_table_info(self, name, namespace):
        cmd = ["python", fate_flow_path, "-f", "table_info", "-t", str(name), "-n", str(namespace)]
        table_info = self.start_task(cmd)
        print(table_info)
        return table_info


class UploadTask(TaskManager):
    def __init__(self, argv=None):
        super().__init__(argv)
        self.method = 'upload'
        self.table_name = None
        self.name_space = None

    def _parse_argv(self, argv):
        role = argv[2]
        if role == GUEST:
            self.party_id = guest_id
        elif role == HOST:
            self.party_id = host_id
        else:
            raise ValueError("Unsupported role:{}".format(role))
        self.role = role
        self.data_file = argv[3]
        if not os.path.exists(self.data_file):
            raise ValueError("file:{} is not found".format(self.data_file))

    def run(self):
        json_info = self._make_upload_conf()
        config = json.dumps(json_info)
        config_path = gen_unique_path(self.method + '_' + self.role)
        config_dir_path = os.path.dirname(config_path)
        os.makedirs(config_dir_path, exist_ok=True)
        with open(config_path, "w") as fout:
            # print("path:{}".format(config_path))
            fout.write(config + "\n")

        print("Upload data config json: {}".format(json_info))
        run_cmd = ['python', fate_flow_path, "-f", "upload", "-c", config_path]
        stdout = self.start_task(run_cmd)
        status = stdout["retcode"]
        if status != 0:
            raise ValueError(
                "upload data exec fail, status:{}, stdout:{}".format(status, stdout))
        print("Upload output is {}".format(stdout))
        time.sleep(6)
        count = self.get_table_info(self.table_name, self.name_space)
        print("Upload Data, role: {}, count: {}".format(self.role, count))
        if self.role == HOST:
            print("The table name and namespace is needed by GUEST. To start a modeling task, please inform "
                  "GUEST with the table name and namespace.")

    def _make_upload_conf(self):
        with open(upload_config_file, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())
        json_info["file"] = self.data_file
        json_info['work_mode'] = work_mode

        time_str = get_timeid()
        self.table_name = '{}_table_name_{}'.format(self.role, time_str)
        self.name_space = '{}_table_namespace_{}'.format(self.role, time_str)

        json_info["table_name"] = self.table_name
        json_info["namespace"] = self.name_space
        print("upload_task, table_name:{}".format(self.table_name))
        print("upload_task, namespace:{}".format(self.name_space))
        return json_info


class TrainTask(TaskManager):
    def __init__(self, argv):
        super().__init__(argv)
        self.method = 'all'
        self.guest_table_name = None
        self.guest_namespace = None

    def _parse_argv(self, argv):
        self._data_type = argv[2]
        self.data_file = argv[3]
        self.host_name = argv[4]
        self.host_namespace = argv[5]
        if self._data_type == 'fast':
            self.task_data_count = 569
            self.task_intersect_count = 569
            self.task_hetero_lr_base_auc = 0.98
        elif self._data_type == "normal":
            self.task_data_count = 30000
            self.task_intersect_count = 30000
            self.task_hetero_lr_base_auc = 0.69
        else:
            raise ValueError("Unknown data type:{}".format(self._data_type))

    def run(self):
        self.guest_table_name, self.guest_namespace = self._upload_data()

        config_dir_path = self._make_runtime_conf()
        start_task_cmd = ['python', fate_flow_path, "-f", "submit_job", "-c",
                          config_dir_path, "-d", hetero_lr_dsl_file]
        stdout = self.start_task(start_task_cmd)
        status = stdout["retcode"]
        if status != 0:
            raise ValueError(
                "Training task exec fail, status:{}, stdout:{}".format(status, stdout))
        else:
            jobid = stdout["jobId"]

        components = self._parse_dsl_components()
        for cpn in components:
            params = [jobid, cpn]
            if "intersect" in cpn:
                max_time = MAX_INTERSECT_TIME
            elif 'lr' in cpn:
                max_time = MAX_TRAIN_TIME
            else:
                max_time = OTHER_TASK_TIME
            job_status = self.start_block_func(self._check_cpn_status, params,
                                               exit_func=self._check_exit, max_waiting_time=max_time)
            print("component name: {}, job_status: {}".format(cpn, job_status))

        auc = self._get_auc(jobid)
        if auc < self.task_hetero_lr_base_auc:
            print("[Warning]  The auc: {} is lower than expect value: {}".format(auc, self.task_hetero_lr_base_auc))
        else:
            print("[Train] train auc:{}".format(auc))

    def _upload_data(self):
        upload_obj = UploadTask()
        upload_obj.role = GUEST
        upload_obj.data_file = self.data_file
        upload_obj.run()
        guest_table_name = upload_obj.table_name
        guest_namespace = upload_obj.name_space
        table_info = self.get_table_info(guest_table_name, guest_namespace)
        count = table_info['data']['count']
        if count != self.task_data_count:
            raise ValueError(
                "[Failed] Test upload task error, upload data count is:{},"
                " it should be:{}".format(count, self.task_data_count))
        else:
            print("Test upload task success, upload count match DTable count")
        return guest_table_name, guest_namespace

    def _make_runtime_conf(self):
        with open(hetero_lr_config_file, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())

        json_info['role']['guest'] = [guest_id]
        json_info['role']['host'] = [host_id]
        json_info['role']['arbiter'] = [arbiter_id]

        json_info['initiator']['party_id'] = guest_id
        json_info['job_parameters']['work_mode'] = work_mode

        table_info = {"name": self.guest_table_name,
                      "namespace": self.guest_namespace}
        json_info["role_parameters"]["guest"]["args"]["data"]["train_data"] = [table_info]

        table_info = {"name": self.host_name,
                      "namespace": self.host_namespace}
        json_info["role_parameters"]["host"]["args"]["data"]["train_data"] = [table_info]
        config = json.dumps(json_info)
        config_path = gen_unique_path('submit_job_guest')
        config_dir_path = os.path.dirname(config_path)
        os.makedirs(config_dir_path, exist_ok=True)
        with open(config_path, "w") as fout:
            # print("path:{}".format(config_path))
            fout.write(config + "\n")
        return config_path

    def _parse_dsl_components(self):
        with open(hetero_lr_dsl_file, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())
        components = list(json_info['components'].keys())
        return components

    def _check_cpn_status(self, job_id, component_name):
        check_cmd = ['python', fate_flow_path, "-f", "query_task", "-j", job_id, "-cpn", component_name]
        stdout = self.start_task(check_cmd)
        try:
            status = stdout["retcode"]
            if status != 0:
                return RUNNING
            # print("In _check_cpn_status, status: {}".format(status))
            task_status = []
            check_data = stdout["data"]

            # Collect all party status
            for component_stats in check_data:
                status = component_stats['f_status']
                task_status.append(status)

            print("Current task status: {}".format(task_status))

            if any([s == FAIL for s in task_status]):
                return FAIL
            if any([s == RUNNING for s in task_status]):
                return RUNNING
        except:
            return None
        return SUCCESS

    @staticmethod
    def _check_exit(status):
        if status is None:
            return True

        if status in [RUNNING, START]:
            return False
        return True

    def _get_auc(self, jobid):
        cmd = ["python", fate_flow_path, "-f", "component_metric_all", "-j", jobid, "-p", str(guest_id), "-r",
               GUEST, "-cpn", evaluation_component_name]
        eval_res = self.start_block_task(cmd, max_waiting_time=OTHER_TASK_TIME)
        eval_results = eval_res['data']['train'][train_component_name]['data']
        auc = 0
        for metric_name, metric_value in eval_results:
            if metric_name == 'auc':
                auc = metric_value
        return auc


class DeleteTableTask(TaskManager):
    def _parse_argv(self, argv):
        pass


if __name__ == "__main__":
    method = sys.argv[1]
    if method == "upload":
        task_obj = UploadTask(sys.argv)
    else:
        task_obj = TrainTask(sys.argv)
    task_obj.run()
