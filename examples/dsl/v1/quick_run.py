#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback

HOME_DIR = os.path.split(os.path.realpath(__file__))[0]
CONFIG_DIR = '/'.join([HOME_DIR, 'config'])
FATE_FLOW_PATH = HOME_DIR + "/../../fate_flow/fate_flow_client.py"
UPLOAD_PATH = HOME_DIR + "/upload_data.json"

GUEST = 'guest'
HOST = 'host'

# You can set up your own configuration files here
DSL_PATH = 'hetero_logistic_regression/test_hetero_lr_train_job_dsl.json'
SUBMIT_CONF_PATH = 'hetero_logistic_regression/test_hetero_lr_train_job_conf.json'

# DSL_PATH = 'homo_logistic_regression/test_homolr_train_job_dsl.json'
# SUBMIT_CONF_PATH = 'homo_logistic_regression/test_homolr_train_job_conf.json'

TEST_PREDICT_CONF = HOME_DIR + '/test_predict_conf.json'

# Define what type of task it is
TASK = 'train'
# TASK = 'predict'

# Put your data to /examples/data folder and indicate the data names here
GUEST_DATA_SET = 'breast_hetero_guest.csv'
HOST_DATA_SET = 'breast_hetero_host.csv'
# GUEST_DATA_SET = 'default_credit_homo_guest.csv'
# HOST_DATA_SET = 'default_credit_homo_host.csv'


# Define your party ids here
GUEST_ID = 10000
HOST_ID = 10000
ARBITER_ID = 10000

# 0 represent for standalone version while 1 represent for cluster version
WORK_MODE = 0

# Time out for waiting a task
MAX_WAIT_TIME = 3600

# Time interval for querying task status
RETRY_JOB_STATUS_TIME = 10

# Your task status list
SUCCESS = 'success'
RUNNING = 'running'
FAIL = 'failed'

# Your latest trained model info is stored here and this will be used when starting a predict task
LATEST_TRAINED_RESULT = '/'.join([HOME_DIR, 'user_config', 'train_info.json'])


def get_timeid():
    return str(int(time.time())) + "_" + str(random.randint(1000, 9999))


def gen_unique_path(prefix):
    return HOME_DIR + "/user_config/" + prefix + ".config_" + get_timeid()


def prettify(response, verbose=True):
    if verbose:
        print(json.dumps(response, indent=4))
        print()
    return response


def save_config_file(config_dict, prefix):
    config = json.dumps(config_dict)
    config_path = gen_unique_path(prefix)
    config_dir_path = os.path.dirname(config_path)
    os.makedirs(config_dir_path, exist_ok=True)
    with open(config_path, "w") as fout:
        # print("path:{}".format(config_path))
        fout.write(config + "\n")
    return config_path


def exec_upload_task(config_dict, role):
    prefix = '_'.join(['upload', role])
    config_path = save_config_file(config_dict=config_dict, prefix=prefix)

    subp = subprocess.Popen(["python",
                             FATE_FLOW_PATH,
                             "-f",
                             "upload",
                             "-c",
                             config_path,
                             "-drop",
                             "1"],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    stdout, stderr = subp.communicate()
    stdout = stdout.decode("utf-8")
    print("stdout:" + str(stdout))
    stdout = json.loads(stdout)
    status = stdout["retcode"]
    if status != 0:
        raise ValueError(
            "[Upload task]exec fail, status:{}, stdout:{}".format(status, stdout))
    return stdout


def exec_modeling_task(dsl_dict, config_dict):
    dsl_path = save_config_file(dsl_dict, 'train_dsl')
    conf_path = save_config_file(config_dict, 'train_conf')
    print("dsl_path: {}, conf_path: {}".format(dsl_path, conf_path))
    subp = subprocess.Popen(["python",
                             FATE_FLOW_PATH,
                             "-f",
                             "submit_job",
                             "-c",
                             conf_path,
                             "-d",
                             dsl_path
                             ],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    stdout, stderr = subp.communicate()
    stdout = stdout.decode("utf-8")
    print("stdout:" + str(stdout))
    stdout = json.loads(stdout)
    with open(LATEST_TRAINED_RESULT, 'w') as outfile:
        json.dump(stdout, outfile)

    status = stdout["retcode"]
    if status != 0:
        raise ValueError(
            "[Trainning Task]exec fail, status:{}, stdout:{}".format(status, stdout))
    return stdout


def job_status_checker(jobid):
    # check_counter = 0
    # while True:
    subp = subprocess.Popen(["python",
                             FATE_FLOW_PATH,
                             "-f",
                             "query_job",
                             "-j",
                             jobid
                             ],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    stdout, stderr = subp.communicate()
    stdout = stdout.decode("utf-8")
    stdout = json.loads(stdout)
    status = stdout["retcode"]
    if status != 0:
        return RUNNING
    check_data = stdout["data"]
    task_status = []

    for component_stats in check_data:
        status = component_stats['f_status']
        task_status.append(status)
    if any([s == FAIL for s in task_status]):
        return FAIL

    if any([s in [RUNNING, 'waiting'] for s in task_status]):
        return RUNNING

    return SUCCESS


def wait_query_job(jobid):
    start = time.time()
    while True:
        job_status = job_status_checker(jobid)
        if job_status == SUCCESS:
            print("Task Finished")
            break
        elif job_status == FAIL:
            print("Task Failed")
            break
        else:
            time.sleep(RETRY_JOB_STATUS_TIME)
            end = time.time()
            print("Task is running, wait time: {}".format(end - start))

            if end - start > MAX_WAIT_TIME:
                print("Task Failed, may by stuck in federation")
                break


def generate_data_info(role):
    if role == GUEST:
        data_name = GUEST_DATA_SET
    else:
        data_name = HOST_DATA_SET

    if '.' in data_name:
        table_name_list = data_name.split('.')
        table_name = '_'.join(table_name_list[:-1])
    else:
        table_name = data_name
    table_name_space = "experiment"
    return table_name, table_name_space


def upload(role):
    with open(UPLOAD_PATH, 'r', encoding='utf-8') as f:
        json_info = json.loads(f.read())

    json_info['work_mode'] = int(WORK_MODE)
    table_name, table_name_space = generate_data_info(role)
    if role == GUEST:
        file_path = 'examples/data/{}'.format(GUEST_DATA_SET)
    else:
        file_path = 'examples/data/{}'.format(HOST_DATA_SET)
    json_info['file'] = file_path
    json_info['table_name'] = table_name
    json_info['namespace'] = table_name_space

    print("Upload data config json: {}".format(json_info))
    stdout = exec_upload_task(json_info, role)
    return stdout


def submit_job():
    with open(DSL_PATH, 'r', encoding='utf-8') as f:
        dsl_json = json.loads(f.read())

    with open(SUBMIT_CONF_PATH, 'r', encoding='utf-8') as f:
        conf_json = json.loads(f.read())

    conf_json['job_parameters']['work_mode'] = WORK_MODE

    conf_json['initiator']['party_id'] = GUEST_ID
    conf_json['role']['guest'] = [GUEST_ID]
    conf_json['role']['host'] = [HOST_ID]
    conf_json['role']['arbiter'] = [ARBITER_ID]

    guest_table_name, guest_namespace = generate_data_info(GUEST)
    host_table_name, host_namespace = generate_data_info(HOST)

    conf_json['role_parameters']['guest']['args']['data']['train_data'] = [
        {
            'name': guest_table_name,
            'namespace': guest_namespace
        }
    ]
    conf_json['role_parameters']['host']['args']['data']['train_data'] = [
        {
            'name': host_table_name,
            'namespace': host_namespace
        }
    ]

    # print("Submit job config json: {}".format(conf_json))
    stdout = exec_modeling_task(dsl_json, conf_json)
    job_id = stdout['jobId']
    fate_board_url = stdout['data']['board_url']
    print("Please check your task in fate-board, url is : {}".format(fate_board_url))
    log_path = HOME_DIR + '/../../logs/{}'.format(job_id)
    print("The log info is located in {}".format(log_path))
    wait_query_job(job_id)


def predict_task():
    try:
        with open(LATEST_TRAINED_RESULT, 'r', encoding='utf-8') as f:
            model_info = json.loads(f.read())
    except FileNotFoundError:
        raise FileNotFoundError('Train Result not Found, please finish a train task before going to predict task')

    model_id = model_info['data']['model_info']['model_id']
    model_version = model_info['data']['model_info']['model_version']

    with open(TEST_PREDICT_CONF, 'r', encoding='utf-8') as f:
        predict_conf = json.loads(f.read())

    predict_conf['initiator']['party_id'] = GUEST_ID
    predict_conf['job_parameters']['work_mode'] = WORK_MODE
    predict_conf['job_parameters']['model_id'] = model_id
    predict_conf['job_parameters']['model_version'] = model_version

    predict_conf['role']['guest'] = [GUEST_ID]
    predict_conf['role']['host'] = [HOST_ID]
    predict_conf['role']['arbiter'] = [ARBITER_ID]

    guest_table_name, guest_namespace = generate_data_info(GUEST)
    host_table_name, host_namespace = generate_data_info(HOST)

    predict_conf['role_parameters']['guest']['args']['data']['validate_data'] = [
        {
            'name': guest_table_name,
            'namespace': guest_namespace
        }
    ]

    predict_conf['role_parameters']['host']['args']['data']['validate_data'] = [
        {
            'name': host_table_name,
            'namespace': host_namespace
        }
    ]

    predict_conf_path = save_config_file(predict_conf, 'predict_conf')
    subp = subprocess.Popen(["python",
                             FATE_FLOW_PATH,
                             "-f",
                             "submit_job",
                             "-c",
                             predict_conf_path
                             ],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    stdout, stderr = subp.communicate()
    stdout = stdout.decode("utf-8")
    print("stdout:" + str(stdout))
    stdout = json.loads(stdout)
    status = stdout["retcode"]
    job_id = stdout['jobId']
    wait_query_job(job_id)
    if status != 0:
        raise ValueError(
            "[Upload task]exec fail, status:{}, stdout:{}".format(status, stdout))
    return stdout


def upload_data():
    if WORK_MODE == 0:
        upload(GUEST)
        time.sleep(3)
        upload(HOST)
        time.sleep(3)
    else:
        if args.role == HOST:
            upload(HOST)
        else:
            upload(GUEST)
            time.sleep(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--role', required=False, type=str, help="role",
                        choices=(GUEST, HOST), default=GUEST
                        )
    try:
        args = parser.parse_args()
        upload_data()
        if args.role == HOST:
            pass
        elif TASK == 'train':
            submit_job()
        else:
            predict_task()

    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        response = {'retcode': 100, 'retmsg': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
        prettify(response)
