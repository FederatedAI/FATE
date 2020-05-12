import json
import os
import random
import subprocess
import sys
import time

home_dir = os.path.split(os.path.realpath(__file__))[0]

# Hetero-lr task
hetero_lr_config_file = home_dir + "/config/test_hetero_lr_train_job_conf.json"
hetero_lr_dsl_file = home_dir + "/config/test_hetero_lr_train_job_dsl.json"

# hetero_lr_config_file = home_dir + "/config/test_hetero_lr_train_job_conf_with_localbaseline.json"
# hetero_lr_dsl_file = home_dir + "/config/test_hetero_lr_train_job_dsl_with_localbaseline.json"

predict_task_file = home_dir + "/config/test_predict_conf.json"

fate_flow_path = home_dir + "/../../../fate_flow/fate_flow_client.py"

# Should be one of "tag_integer_value", "tag", "tag_float_value", "tag_1" or "label"
HOST_DATA_TYPE = 'tag_integer_value'

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
MAX_INTERSECT_TIME = 3600  # In millisecond
MAX_TRAIN_TIME = 7200  # In millisecond
WAIT_UPLOAD_TIME = 1000
OTHER_TASK_TIME = 7200
# RETRY_JOB_STATUS_TIME = 5
STATUS_CHECKER_TIME = 10

# Upload and download data
upload_config_file = home_dir + "/config/upload.json"
download_config_file = home_dir + "/config/download.json"



def get_timeid():
    return str(int(time.time())) + "_" + str(random.randint(1000, 9999))


def gen_unique_path(prefix):
    return home_dir + "/test/" + prefix + ".config_" + get_timeid()


class TaskManager(object):
    def __init__(self, argv=None):
        self.guest_id = 9999
        self.host_id = 10000
        self.arbiter_id = 10000
        self.work_mode = 1
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
            # print("Before enter exit func, result is : {}".format(result))
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
            status = stdout["retcode"]
            if status != 0:
                raise ValueError(
                    "upload data exec fail, status:{}, stdout:{}".format(status, stdout))
        except:
            print("stdout is {}".format(stdout))
        return stdout

    def get_table_info(self, name, namespace):
        cmd = ["python", fate_flow_path, "-f", "table_info", "-t", str(name), "-n", str(namespace)]
        table_info = self.start_task(cmd)
        print(table_info)
        return table_info


class UploadTask(TaskManager):
    def __init__(self, argv=None):
        self.table_name = None
        self.name_space = None
        self.role = None
        self.file_name = None
        super().__init__(argv)
        self.method = 'upload'

    def _parse_argv(self, argv):
        role = argv[2]
        self.work_mode = int(argv[3])
        self.set_role(role, self.work_mode)

    def set_role(self, role, work_mode):
        self.role = role
        self.table_name = 'mork_data_{}'.format(role)
        self.name_space = 'mork_data'
        self.file_name = '/generated_data_{}.csv'.format(self.role)
        self.work_mode = work_mode

    def run(self):
        # from . import generate_mock_data
        if self.role == HOST:
            data_type = HOST_DATA_TYPE
        else:
            data_type = 'label'
        run_cmd = ['python', "generate_mock_data.py", data_type, self.role]
        self.start_task(run_cmd)

        json_info = self._make_upload_conf()
        config = json.dumps(json_info)
        config_path = gen_unique_path(self.method + '_' + self.role)
        config_dir_path = os.path.dirname(config_path)
        os.makedirs(config_dir_path, exist_ok=True)
        with open(config_path, "w") as fout:
            # print("path:{}".format(config_path))
            fout.write(config + "\n")

        print("Upload data config json: {}".format(json_info))
        run_cmd = ['python', fate_flow_path, "-f", "upload", "-c", config_path, "-drop", "1"]
        stdout = self.start_task(run_cmd)

        print("Upload output is {}".format(stdout))
        time.sleep(WAIT_UPLOAD_TIME / 100)
        count = self.get_table_info(self.table_name, self.name_space)
        print("Upload Data, role: {}, count: {}".format(self.role, count))

    def _make_upload_conf(self):
        with open(upload_config_file, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())
        json_info["file"] = home_dir + self.file_name
        json_info['work_mode'] = self.work_mode

        json_info["table_name"] = self.table_name
        json_info["namespace"] = self.name_space
        if self.role == HOST:
            json_info["head"] = 0
        print("upload_task, table_name:{}".format(self.table_name))
        print("upload_task, namespace:{}".format(self.name_space))
        return json_info


class TrainTask(TaskManager):
    def __init__(self, argv):
        super().__init__(argv)
        self.method = 'all'
        self.guest_table_name = None
        self.guest_namespace = None
        self.model_id = None
        self.model_version = None

    def _parse_argv(self, argv):
        self.role = argv[2]
        self.work_mode = int(argv[3])
        self.guest_id = int(argv[4])
        self.host_id = int(argv[5])
        self.arbiter_id = int(argv[6])

    def run(self):
        self.guest_table_name, self.guest_namespace = self._upload_data()

        config_dir_path = self._make_runtime_conf()
        start_task_cmd = ['python', fate_flow_path, "-f", "submit_job", "-c",
                          config_dir_path, "-d", hetero_lr_dsl_file]
        stdout = self.start_task(start_task_cmd)
        print("Job has started, stdout is : {}".format(stdout))
        assert isinstance(stdout, dict)
        status = stdout["retcode"]
        self.model_id = stdout['data']['model_info']['model_id']
        self.model_version = stdout['data']['model_info']['model_version']
        if status != 0:
            raise ValueError(
                "Training task exec fail, status:{}, stdout:{}".format(status, stdout))
        else:
            jobid = stdout["jobId"]
        #
        components = self._parse_dsl_components()
        num_of_components = len(components)

        max_time = OTHER_TASK_TIME * num_of_components
        job_status = self.start_block_func(self._check_cpn_status, [jobid],
                                           exit_func=self._check_exit, max_waiting_time=max_time)
        # # print("component name: {}, job_status: {}".format(cpn, job_status))
        # # if job_status == FAIL:
        # #     exit(1)
        #
        auc = self._get_auc(jobid)
        print("[Train] train auc:{}".format(auc))
        time.sleep(WAIT_UPLOAD_TIME / 100)
        self.start_predict_task()

    def start_predict_task(self):
        config_dir_path = self._make_runtime_conf("predict")
        start_task_cmd = ['python', fate_flow_path, "-f", "submit_job", "-c", config_dir_path]
        stdout = self.start_task(start_task_cmd)
        assert isinstance(stdout, dict)
        status = stdout["retcode"]
        if status != 0:
            raise ValueError(
                "Training task exec fail, status:{}, stdout:{}".format(status, stdout))
        else:
            jobid = stdout["jobId"]

        components = self._parse_dsl_components()
        num_of_components = len(components)

        max_time = OTHER_TASK_TIME * num_of_components
        job_status = self.start_block_func(self._check_cpn_status, [jobid],
                                           exit_func=self._check_exit, max_waiting_time=max_time)
        if job_status == FAIL:
            exit(1)
        print("[Predict Task] Predict success")

    def _upload_data(self):
        upload_obj = UploadTask()
        # upload_obj.role = GUEST
        upload_obj.set_role(GUEST, self.work_mode)
        upload_obj.run()
        guest_table_name = upload_obj.table_name
        guest_namespace = upload_obj.name_space
        time.sleep(WAIT_UPLOAD_TIME / 100)
        table_info = self.get_table_info(guest_table_name, guest_namespace)
        assert isinstance(table_info, dict)
        count = table_info['data']['count']
        print("Test upload task success, upload count is {}".format(count))
        return guest_table_name, guest_namespace

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

        table_info = {"name": self.guest_table_name.replace('guest', 'host'),
                      "namespace": self.guest_namespace.replace('guest', 'host')}
        if conf_type == 'train':
            json_info["role_parameters"]["host"]["args"]["data"]["train_data"] = [table_info]
            json_info["role_parameters"]["host"]["args"]["data"]["eval_data"] = [table_info]
            if HOST_DATA_TYPE in ["tag_float_value", "tag"]:
                json_info["role_parameters"]["host"]["dataio_0"]["delimitor"] = [";"]
            else:
                json_info["role_parameters"]["host"]["dataio_0"]["delimitor"] = [","]

            if HOST_DATA_TYPE in ["tag"]:
                json_info["role_parameters"]["host"]["dataio_0"]["tag_with_value"] = [False]
            else:
                json_info["role_parameters"]["host"]["dataio_0"]["tag_with_value"] = [True]

        else:
            json_info["role_parameters"]["host"]["args"]["data"]["eval_data"] = [table_info]

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

    def _check_cpn_status(self, job_id):
        check_cmd = ['python', fate_flow_path, "-f", "query_job", "-j", job_id]
        stdout = self.start_task(check_cmd)
        assert isinstance(stdout, dict)
        try:
            # print("In _check_cpn_status, stdout is : {}".format(stdout))
            status = stdout["retcode"]
            if status != 0:
                return RUNNING
            # print("In _check_cpn_status, status: {}".format(status))
            task_status = []
            check_data = stdout["data"]
            current_tasks = []
            # Collect all party status
            for component_stats in check_data:
                status = component_stats['f_status']
                current_task = component_stats['f_current_tasks']
                task_status.append(status)
                current_tasks.append(current_task)
            print("Current task: {}, status: {}".format(current_tasks, task_status))

            if any([s == FAIL for s in task_status]):
                return FAIL
            if any([s == RUNNING for s in task_status]):
                return RUNNING
        except:
            return None
        return SUCCESS

    @staticmethod
    def _check_exit(status):
        # if status is None:
        #     return True
        if status == SUCCESS:
            return True
        return False

    def _get_auc(self, jobid):
        cmd = ["python", fate_flow_path, "-f", "component_metric_all", "-j", jobid, "-p", str(self.guest_id), "-r",
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
