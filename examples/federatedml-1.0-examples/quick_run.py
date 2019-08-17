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
UPLOAD_GUEST = HOME_DIR + "/upload_data_guest.json"
UPLOAD_HOST = HOME_DIR + "/upload_data_host.json"

GUEST = 'guest'
HOST = 'host'

HETERO_LR = 'hetero_lr'
HOMO_LR = 'homo_lr'
HETERO_SECUREBOOST = 'hetero_secureboost'

ALGORITHM_LIST = (HETERO_LR, HOMO_LR, HETERO_SECUREBOOST)


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
                             config_path],
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
    status = stdout["retcode"]
    if status != 0:
        raise ValueError(
            "[Trainning Task]exec fail, status:{}, stdout:{}".format(status, stdout))
    return stdout


def upload(work_mode, role):
    if role == GUEST:
        with open(UPLOAD_GUEST, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())
    else:
        with open(UPLOAD_HOST, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())

    json_info['work_mode'] = int(work_mode)

    print("Upload data config json: {}".format(json_info))
    stdout = exec_upload_task(json_info, role)
    return stdout


def upload_homo(work_mode, role):
    if role == GUEST:
        with open(UPLOAD_GUEST, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())
    else:
        with open(UPLOAD_HOST, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())

    if role == GUEST:
        json_info['file'] = 'examples/data/breast_homo_guest.csv'
        json_info['table_name'] = 'homo_breast_guest'
        json_info['namespace'] = 'homo_breast_guest'
    else:
        json_info['file'] = 'examples/data/breast_homo_host.csv'
        json_info['table_name'] = 'homo_breast_host'
        json_info['namespace'] = 'homo_breast_host'

    json_info['work_mode'] = int(work_mode)
    print("Upload data config json: {}".format(json_info))
    stdout = exec_upload_task(json_info, role)
    return stdout


def submit_job(algorithm, work_mode, guest_id, host_id, arbiter_id):

    if algorithm is None or algorithm == HETERO_LR:
        dsl_path = '/'.join([HOME_DIR, 'hetero_logistic_regression', 'test_hetero_lr_train_job_dsl.json'])
        conf_path = '/'.join([HOME_DIR, 'hetero_logistic_regression', 'test_hetero_lr_train_job_conf.json'])
    elif algorithm == HOMO_LR:
        dsl_path = '/'.join([HOME_DIR, 'homo_logistic_regression', 'test_homolr_train_job_dsl.json'])
        conf_path = '/'.join([HOME_DIR, 'homo_logistic_regression', 'test_homolr_train_job_conf.json'])
    elif algorithm == HETERO_SECUREBOOST:
        dsl_path = '/'.join([HOME_DIR, 'hetero_secureboost', 'test_secureboost_train_dsl.json'])
        conf_path = '/'.join([HOME_DIR, 'hetero_secureboost', 'test_secureboost_train_binary_conf.json'])
    else:
        raise ValueError("Unsupported algorithm: {}, should be ".format(algorithm))

    with open(dsl_path, 'r', encoding='utf-8') as f:
        dsl_json = json.loads(f.read())

    with open(conf_path, 'r', encoding='utf-8') as f:
        conf_json = json.loads(f.read())

    if work_mode is not None:
        conf_json['job_parameters']['work_mode'] = int(work_mode)

    if guest_id is not None:
        conf_json['initiator']['party_id'] = guest_id
        conf_json['role']['guest'] = [int(guest_id)]

    if host_id is not None:
        conf_json['role']['host'] = [int(host_id)]

    if arbiter_id is not None:
        conf_json['role']['arbiter'] = [int(arbiter_id)]

    # print("Submit job config json: {}".format(conf_json))
    stdout = exec_modeling_task(dsl_json, conf_json)
    job_id = stdout['jobId']
    fate_board_url = stdout['data']['board_url']
    print("Please check your task in fate-board, url is : {}".format(fate_board_url))
    log_path = HOME_DIR + '/../../logs/{}'.format(job_id)
    print("The log info is located in {}".format(log_path))


def parameter_checker(args):
    if args.work_mode is not None:
        if args.work_mode not in ['0', '1']:
            raise ValueError('Unsupported Work mode parameter {} should be 0 or 1'.format(args.work_mode))
        this_work_mode = int(args.work_mode)
    else:
        this_work_mode = 0

    if args.algorithm is not None:
        if args.algorithm not in ALGORITHM_LIST:
            raise ValueError('Unsupported algorithm parameter {} should be one of: {}'.format(
                args.work_mode, ALGORITHM_LIST))

    if args.role is not None:
        if args.role not in [GUEST, HOST]:
            raise ValueError('Unsupported role parameter {} should be one of: {}'.format(
                args.work_mode, [GUEST, HOST]))

    if this_work_mode == 1 and args.role is None:
        raise ValueError("In cluster version please indicate role")

    return this_work_mode


def cluster_host(args):
    if args.algorithm == HOMO_LR:
        upload_homo(work_mode=1, role=HOST)
    else:
        upload(work_mode=1, role=HOST)


def cluster_guest(args):
    if args.algorithm == HOMO_LR:
        upload_homo(work_mode=1, role=GUEST)
    else:
        upload(work_mode=1, role=GUEST)
    time.sleep(3)
    submit_job(args.algorithm, args.work_mode, args.guest_party_id, args.host_party_id, args.arbiter_party_id)


def standalone(args):
    if args.algorithm == HOMO_LR:
        upload_func = upload_homo
    else:
        upload_func = upload

    upload_func(0, GUEST)
    time.sleep(3)
    upload_func(0, HOST)
    time.sleep(3)
    submit_job(args.algorithm, args.work_mode, args.guest_party_id, args.host_party_id, args.arbiter_party_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gid', '--guest_party_id', required=False, type=str, help="guest party id")
    parser.add_argument('-hid', '--host_party_id', required=False, type=str, help="host party id")
    parser.add_argument('-aid', '--arbiter_party_id', required=False, type=str, help="arbiter party id")
    parser.add_argument('-w', '--work_mode', required=False, type=str, help="work mode", default='0')
    parser.add_argument('-r', '--role', required=False, type=str, help="role")
    parser.add_argument('-a', '--algorithm', required=False, type=str, help="algorithm type",
                        choices=ALGORITHM_LIST, default=HETERO_LR)

    try:
        args = parser.parse_args()
        work_mode = parameter_checker(args)
        if work_mode == 0:
            standalone(args)
        else:
            if args.role == HOST:
                cluster_host(args)
            else:
                cluster_guest(args)

    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        response = {'retcode': 100, 'retmsg': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
        prettify(response)
