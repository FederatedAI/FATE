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

# Intersect task
intersect_dsl_file = home_dir + "/config/test_intersect_job_dsl.json"
intersect_conf_file = home_dir + "/config/test_intersect_job_conf.json"

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
RETRY_JOB_STATUS_TIME = 5
WORKFLOW_STATUS_CHECKER_TIME = 5

TEST_TASK = {'TEST_UPLOAD': 2, 'TEST_INTERSECT': 2, 'TEST_TRAIN': 2}


def get_timeid():
    return str(int(time.time())) + "_" + str(random.randint(1000, 9999))


def gen_unique_path(prefix):
    return home_dir + "/test/" + prefix + ".config_" + get_timeid()


def exec_task(config_dict, task, role, dsl_path=None):
    config = json.dumps(config_dict)
    config_path = gen_unique_path(task + '_' + role)
    config_dir_path = os.path.dirname(config_path)
    os.makedirs(config_dir_path, exist_ok=True)
    with open(config_path, "w") as fout:
        # print("path:{}".format(config_path))
        fout.write(config + "\n")

    # For upload, download task
    if dsl_path is None:
        subp = subprocess.Popen(["python",
                                 fate_flow_path,
                                 "-f",
                                 task,
                                 "-c",
                                 config_path],
                                shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    # For submit_job task
    else:
        subp = subprocess.Popen(["python",
                                 fate_flow_path,
                                 "-f",
                                 task,
                                 "-d",
                                 dsl_path,
                                 "-c",
                                 config_path],
                                shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

    stdout, stderr = subp.communicate()
    # subp.wait()
    # print("Current subp status: {}".format(subp.returncode))
    stdout = stdout.decode("utf-8")
    print("stdout:" + str(stdout))
    stdout = json.loads(stdout)
    status = stdout["retcode"]
    if status != 0:
        raise ValueError(
            "[exec_task] task:{}, role:{} exec fail, status:{}, stdout:{}".format(task, role, status, stdout))

    return stdout


def obtain_component_output(jobid, party_id, role, component_name, output_type='data'):
    task_type = 'component_output_data'
    if output_type == 'data':
        task_type = 'component_output_data'
        cmd = ["python",
               fate_flow_path,
               "-f",
               task_type,
               "-j",
               jobid,
               "-p",
               str(party_id),
               "-r",
               role,
               "-cpn",
               component_name,
               "-o",
               home_dir
               ]
    elif output_type == 'model':
        task_type = 'component_output_model'
        cmd = ["python",
               fate_flow_path,
               "-f",
               task_type,
               "-j",
               jobid,
               "-p",
               str(party_id),
               "-r",
               role,
               "-cpn",
               component_name
               ]
    elif output_type == 'log_metric':
        task_type = 'component_metric_all'
        cmd = ["python",
               fate_flow_path,
               "-f",
               task_type,
               "-j",
               jobid,
               "-p",
               str(party_id),
               "-r",
               role,
               "-cpn",
               component_name
               ]
    else:
        cmd = []

    retry_counter = 0
    while True:
        print("exec cmd: {}".format(cmd))
        subp = subprocess.Popen(cmd,
                                shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        stdout, stderr = subp.communicate()
        # subp.wait()
        stdout = stdout.decode("utf-8")
        if not stdout:

            retry_counter += 1
            if retry_counter >= 5:
                raise ValueError(
                    "[obtain_component_output] task:{} failed stdout:{}".format(task_type, stdout))
            time.sleep(5)
        else:
            break

    print("task_type: {}, jobid: {}, party_id: {}, role: {}, component_name: {}".format(
        task_type, job_id, party_id, role, component_name
    ))

    print("obtain_component_output stdout:" + str(stdout))
    stdout = json.loads(stdout)
    return stdout


def parse_exec_task(stdout):
    parse_result = {}
    try:
        parse_result["table_name"] = stdout["data"]["table_name"]
    except:
        parse_result["table_name"] = None

    try:
        parse_result["namespace"] = stdout["data"]["namespace"]
    except:
        parse_result["namespace"] = None

    parse_result["jobId"] = stdout["jobId"]

    return parse_result


def job_status_checker(jobid, component_name):
    check_counter = 0
    while True:
        subp = subprocess.Popen(["python",
                                 fate_flow_path,
                                 "-f",
                                 "query_task",
                                 "-j",
                                 jobid,
                                 "-cpn",
                                 component_name
                                 ],
                                shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        stdout, stderr = subp.communicate()
        # subp.wait()
        # print("Current subp status: {}".format(subp.returncode))
        stdout = stdout.decode("utf-8")
        # print("Job_status_checker Stdout is : {}".format(stdout))
        stdout = json.loads(stdout)
        status = stdout["retcode"]
        if status != 0:
            if check_counter >= 60:
                raise ValueError("jobid:{} status exec fail, status:{}".format(jobid, status))
            print("Current retry times: {}".format(check_counter))
            time.sleep(10)
            check_counter += 1
        else:
            break
            #     raise ValueError("jobid:{} status exec fail, status:{}".format(jobid, status))

    return stdout


def upload(config_file, self_party_id, role, data_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        json_info = json.loads(f.read())
    json_info["file"] = data_file
    json_info["local"]["party_id"] = self_party_id
    json_info["local"]["role"] = role
    json_info['work_mode'] = work_mode

    time_str = get_timeid()
    this_table_name = '{}_table_name_{}'.format(role, time_str)
    this_table_namespace = '{}_table_namespace_{}'.format(role, time_str)

    json_info["table_name"] = this_table_name
    json_info["namespace"] = this_table_namespace

    print("Upload data config json: {}".format(json_info))
    stdout = exec_task(json_info, "upload", role)
    print("Upload output is {}".format(stdout))
    parse_result = parse_exec_task(stdout)
    return parse_result["table_name"], parse_result["namespace"]


def download(config_file, self_party_id, this_role, this_table_name, this_table_namespace):
    # write new json
    with open(config_file, 'r', encoding='utf-8') as f:
        json_info = json.loads(f.read())

    json_info['local']['party_id'] = self_party_id
    json_info['local']['role'] = this_role

    json_info['table_name'] = this_table_name
    json_info['namespace'] = this_table_namespace

    stdout = exec_task(json_info, "download", this_role)
    parse_result = parse_exec_task(stdout)
    print("[Task Download] finish download, table_name:{}, namespace:{}".format(
        parse_result["table_name"], parse_result["namespace"]))
    return parse_result["table_name"], parse_result["namespace"]


def download_id_library(config_file, guest_id, host_id, role):
    if role == GUEST:
        self_party_id = guest_id
    elif role == HOST:
        self_party_id = host_id
    else:
        raise ValueError("Unsupport role:{}".format(role))

    # write new json
    json_file = open(config_file, 'r', encoding='utf-8')
    json_info = json.load(json_file)

    json_info['local']['party_id'] = self_party_id
    json_info['local']['role'] = role
    json_info['role']['guest'] = [guest_id]
    json_info['role']['host'] = [host_id]

    print(json_info)
    stdout = exec_task(json_info, "download", role)
    parse_result = parse_exec_task(stdout)
    print("[Task Download]finsih download id libraray data_type:data_input, table_name:{}, namespace:{}".format(
        parse_result["table_name"], parse_result["namespace"]))
    return parse_result["table_name"], parse_result["namespace"]


def task_status_checker(jobid, component_name):
    stdout = job_status_checker(jobid, component_name)
    check_data = stdout["data"]

    retry_counter = 0

    # Wait for task start
    while check_data is None:
        time.sleep(2)
        print("[Workflow_Job_Status_checker] check jobid:{}, status,"
              " current retry counter:{}".format(jobid,
                                                 retry_counter))
        stdout = job_status_checker(jobid, component_name)
        print("In task_status_checker function, retry_counter: {}".format(retry_counter))
        check_data = stdout["data"]
        retry_counter += 1
        if retry_counter >= 5:
            print("[Workflow_Job_Status_checker] retry time >= 5, check jobid failed")
            return FAIL

    task_status = []
    # Collect all party status
    for component_stats in check_data:
        status = component_stats['f_status']
        task_status.append(status)

    print("Current task status: {}".format(task_status))

    for s in task_status:
        if s == FAIL:
            print("return status is fail")
            return FAIL

        if s == RUNNING:
            print("return status is running")
            return RUNNING
    print("return status is success")
    return SUCCESS


def intersect(dsl_file, config_file, guest_id, host_id, guest_name, guest_namespace, host_name, host_namespace):
    # write new json
    with open(config_file, 'r', encoding='utf-8') as f:
        json_info = json.loads(f.read())

    json_info['role']['guest'] = [guest_id]
    json_info['role']['host'] = [host_id]

    json_info['initiator']['party_id'] = guest_id
    json_info['job_parameters']['work_mode'] = work_mode

    table_info = {"name": guest_name,
                  "namespace": guest_namespace}
    json_info["role_parameters"]["guest"]["args"]["data"]["data"] = [table_info]

    table_info = {"name": host_name,
                  "namespace": host_namespace}
    json_info["role_parameters"]["host"]["args"]["data"]["data"] = [table_info]

    start = time.time()
    stdout = exec_task(json_info, "submit_job", "guest_intersect", dsl_path=dsl_file)
    jobid = parse_exec_task(stdout)["jobId"]

    cur_job_status = RUNNING
    workflow_job_status_counter = 0
    while cur_job_status == RUNNING or cur_job_status == START:
        time.sleep(WORKFLOW_STATUS_CHECKER_TIME)
        print("[Intersect] Start intersect job status checker, status counter: {},"
              " jobid:{}".format(workflow_job_status_counter, jobid))
        cur_job_status = task_status_checker(jobid, component_name='intersect_0')
        print("[Intersect] cur job status:{}".format(cur_job_status))
        end = time.time()
        if end - start > MAX_INTERSECT_TIME:
            print("[Intersect] reach max intersect time:{}, intersect task may be failed, and exit now")
            break

        workflow_job_status_counter += 1

    # Wait for Status checker
    # time.sleep(15)

    return cur_job_status, jobid


def get_module_auc(evaluation_result):
    result = evaluation_result.get('train').get(train_component_name).get('data')
    module_auc = None
    for eval_metric in result:
        name = eval_metric[0]
        if name == 'auc':
            module_auc = eval_metric[1]
            break
    return module_auc


def train(dsl_file, config_file, guest_id, host_id, arbiter_id, guest_name, guest_namespace, host_name, host_namespace):
    with open(config_file, 'r', encoding='utf-8') as f:
        json_info = json.loads(f.read())

    json_info['role']['guest'] = [guest_id]
    json_info['role']['host'] = [host_id]
    json_info['role']['arbiter'] = [arbiter_id]

    json_info['initiator']['party_id'] = guest_id
    json_info['job_parameters']['work_mode'] = work_mode

    table_info = {"name": guest_name,
                  "namespace": guest_namespace}
    json_info["role_parameters"]["guest"]["args"]["data"]["train_data"] = [table_info]

    table_info = {"name": host_name,
                  "namespace": host_namespace}
    json_info["role_parameters"]["host"]["args"]["data"]["train_data"] = [table_info]

    start = time.time()
    stdout = exec_task(json_info, "submit_job", "guest_train", dsl_path=dsl_file)
    jobid = parse_exec_task(stdout)["jobId"]

    cur_job_status = RUNNING
    while cur_job_status == RUNNING or cur_job_status == START:
        time.sleep(WORKFLOW_STATUS_CHECKER_TIME)
        cur_job_status = task_status_checker(jobid, evaluation_component_name)
        print("[Train] cur job status:{}, jobid:{}".format(cur_job_status, jobid))
        end = time.time()
        if end - start > MAX_TRAIN_TIME:
            print("[Train] reach max train time:{}, intersect task may be failed, and exit now")
            break
    return cur_job_status, jobid


def get_table_count(name, namespace):
    from arch.api import eggroll
    eggroll.init("get_intersect_output", mode=work_mode)
    table = eggroll.table(name, namespace)
    count = table.count()
    print("table count:{}".format(count))
    return count


def get_table_collect(name, namespace):
    from arch.api import eggroll
    eggroll.init("get_intersect_output", mode=work_mode)
    table = eggroll.table(name, namespace)
    return list(table.collect())


def request_offline_feature(name, namespace, ret_size):
    data_id = get_table_collect(name, namespace)
    ret_ids = []

    i = 0
    while len(ret_ids) < ret_size or i >= len(data_id):
        ret_ids.append(data_id[i][0])
        i += 1

    return ret_ids


def check_file_line_num(file_path):
    file_name = '/'.join([file_path, 'output_data.csv'])
    subp = subprocess.Popen(["wc",
                             "-l",
                             file_name
                             ],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    stdout, stderr = subp.communicate()
    subp.wait()
    print("Current subp status: {}".format(subp.returncode))
    stdout = stdout.decode("utf-8")
    file_length = int(stdout.split()[0])

    print("Job_status_checker Stdout is : {}".format(file_length))
    # stdout = json.loads(stdout)
    return file_length


def split_data_and_save_file(guest_table_name, guest_namespace, host_id, train_file, predict_file):
    guest_data = get_table_collect(guest_table_name, guest_namespace)
    pos_label_set = []
    neg_label_set = []

    print("host_id:{}".format(len(host_id)))
    print("guest_id:{}".format(len(guest_data)))
    for data in guest_data:
        if data[0] in host_id:
            label = data[1].split(',')[0]
            if label == '0':
                neg_label_set.append(data)
            elif label == '1':
                pos_label_set.append(data)
            else:
                raise ValueError("Unknown label:{}".format(label))
        else:
            print("not in host_id:{}".format(data))
    print("pos count:{}".format(len(pos_label_set)))
    print("neg count:{}".format(len(neg_label_set)))

    random.shuffle(pos_label_set)
    random.shuffle(neg_label_set)

    train_pos_size = int(0.8 * len(pos_label_set))
    train_neg_size = int(0.8 * len(neg_label_set))
    train_data = pos_label_set[:train_pos_size]
    predict_data = pos_label_set[train_pos_size:]

    train_data.extend(neg_label_set[:train_neg_size])
    predict_data.extend(neg_label_set[train_neg_size:])

    with open(train_file, 'w') as fout:
        for data in train_data:
            fout.write(data[0] + ',' + data[1] + '\n')

    with open(predict_file, 'w') as fout:
        for data in predict_data:
            fout.write(data[0] + ',' + data[1] + '\n')


if __name__ == "__main__":
    method = sys.argv[1]

    if method == "upload":
        role = sys.argv[2]
        data_file = sys.argv[3]

        if role == GUEST:
            self_party_id = guest_id
        elif role == HOST:
            self_party_id = host_id
        else:
            raise ValueError("Unsupported role:{}".format(role))

        if not os.path.exists(data_file):
            raise ValueError("file:{} is not found".format(data_file))

        table_name, table_namespace = upload(upload_config_file, self_party_id, role, data_file)
        print("table_name:{}".format(table_name))
        print("namespace:{}".format(table_namespace))
        time.sleep(6)
        print("method:{}, count:{}".format(method, get_table_count(table_name, table_namespace)))
        if role == HOST:
            print("The table name and namespace is needed by GUEST. To start a modeling task, please inform "
                  "GUEST with the table name and namespace.")

    elif method == "all":
        task = sys.argv[2]
        data_file = sys.argv[3]
        host_name = sys.argv[4]
        host_namespace = sys.argv[5]

        if task == "fast":
            task_data_count = 569
            task_intersect_count = 569
            task_hetero_lr_base_auc = 0.98
        elif task == "normal":
            task_data_count = 30000
            task_intersect_count = 30000
            task_hetero_lr_base_auc = 0.69
        else:
            raise ValueError("Unknown task:{}".format(task))

        # Upload Data
        print("Start Upload Data")
        table_name, table_namespace = upload(upload_config_file, guest_id, 'guest', data_file)
        print("table_name:{}".format(table_name))
        print("namespace:{}".format(table_namespace))
        time.sleep(6)
        print("Data uploaded, expected table count: {}".format(task_data_count))

        # Download the uploaded data. Check if download
        # guest_table_name, guest_namespace = download(download_config_file, guest_id, "guest",
        #                                              table_name, table_namespace)

        count = get_table_count(table_name, table_namespace)
        if count != task_data_count:
            TEST_TASK["TEST_UPLOAD"] = 1
            raise ValueError(
                "[failed] Test upload intersect task error, upload data count is:{}, it should be:{}".format(count,
                                                                                                             task_data_count))
        else:
            print("Test upload task success, upload count match DTable count")
            TEST_TASK["TEST_UPLOAD"] = 0

        print("[Intersect] Start intersect task")
        job_status, job_id = intersect(intersect_dsl_file,
                                       intersect_conf_file,
                                       guest_id=guest_id,
                                       host_id=host_id,
                                       guest_name=table_name,
                                       guest_namespace=table_namespace,
                                       host_name=host_name,
                                       host_namespace=host_namespace)

        if job_status is SUCCESS:
            print("[Intersect] intersect task status is success")
            intersect_result = obtain_component_output(jobid=job_id,
                                                       role='guest',
                                                       party_id=guest_id,
                                                       component_name='intersect_0',
                                                       output_type='data')
            print("intersect result:{}".format(intersect_result))
            intersect_file_name = intersect_result.get('directory')
            count = check_file_line_num(intersect_file_name)

            # TODO: Wait for fate-flow interface
            # if count == 100:
            #     count = task_intersect_count

            if count != task_intersect_count:
                TEST_TASK["TEST_INTERSECT"] = 1
                raise ValueError(
                    "[failed] Test intersect task error, intersect output count is:{}, it should be:{}".format(count,
                                                                                                               task_intersect_count))
            else:
                TEST_TASK["TEST_INTERSECT"] = 0
        else:
            raise ValueError("intersect task is failed")

        print("[Train] Start train task")
        job_status, job_id = train(dsl_file=hetero_lr_dsl_file,
                                   config_file=hetero_lr_config_file,
                                   guest_id=guest_id,
                                   host_id=host_id,
                                   arbiter_id=arbiter_id,
                                   guest_name=table_name,
                                   guest_namespace=table_namespace,
                                   host_name=host_name,
                                   host_namespace=host_namespace
                                   )

        if job_status is SUCCESS:
            print("[Train] train task status is success")
            eval_res = obtain_component_output(jobid=job_id,
                                               role='guest',
                                               party_id=guest_id,
                                               component_name=evaluation_component_name,
                                               output_type='log_metric')
            eval_results = eval_res['data']['train'][train_component_name]['data']
            auc = 0
            for metric_name, metric_value in eval_results:
                if metric_name == 'auc':
                    auc = metric_value
            print("[Train] train eval:{}".format(eval_results))
            # eval_res = get_table_collect(eval_output_name, eval_output_namespace)
            if auc > task_hetero_lr_base_auc:
                TEST_TASK["TEST_TRAIN"] = 0
        else:
            print("[Train] train task is failed")
            TEST_TASK["TEST_TRAIN"] = 1

        test_success = 0
        test_failed = 0
        for key in TEST_TASK:
            if TEST_TASK[key] == 0:
                print("{} is success".format(key))
                test_success += 1
            else:
                print("{} is failed".format(key))
                test_failed += 1

        print("Test success:{}, failed:{}".format(test_success, test_failed))
