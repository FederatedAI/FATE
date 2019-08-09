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

import os
import subprocess
import sys
import time
import json

home_dir = os.path.split(os.path.realpath(__file__))[0]
fate_flow_path = home_dir + "/../../fate_flow/fate_flow_client.py"


def test_query_job(check_job_id):
    subp = subprocess.Popen(["python",
                             fate_flow_path,
                             "-f",
                             "query_task",
                             "-j",
                             check_job_id
                             ],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    subp.wait()
    print("Current subp status: {}".format(subp.returncode))
    stdout = subp.stdout.read().decode("utf-8")
    print("[test_query_job] Stdout is : {}".format(stdout))
    stdout = json.loads(stdout)
    status = stdout["retcode"]
    if status != 0:
        return False

    return True


def test_get_job_log(check_job_id):
    subp = subprocess.Popen(["python",
                             fate_flow_path,
                             "-f",
                             "job_log",
                             "-j",
                             check_job_id
                             ],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    subp.wait()
    print("Current subp status: {}".format(subp.returncode))
    stdout = subp.stdout.read().decode("utf-8")
    print("[test_get_job_log] Stdout is : {}".format(stdout))
    stdout = json.loads(stdout)
    status = stdout["retcode"]
    if status != 0:
        return False
    return True


def test_job_config(check_job_id):
    subp = subprocess.Popen(["python",
                             fate_flow_path,
                             "-f",
                             "job_config",
                             "-j",
                             check_job_id
                             ],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    subp.wait()
    print("Current subp status: {}".format(subp.returncode))
    stdout = subp.stdout.read().decode("utf-8")
    print("[test_job_config] Stdout is : {}".format(stdout))
    stdout = json.loads(stdout)
    status = stdout["retcode"]
    if status != 0:
        return False
    return True


def check_component_metric_all(check_job_id):
    subp = subprocess.Popen(["python",
                             fate_flow_path,
                             "-f",
                             "component_metric_all",
                             "-j",
                             check_job_id
                             ],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    subp.wait()
    print("Current subp status: {}".format(subp.returncode))
    stdout = subp.stdout.read().decode("utf-8")
    print("[test_job_config] Stdout is : {}".format(stdout))
    stdout = json.loads(stdout)
    status = stdout["retcode"]
    if status != 0:
        return False
    return True


if __name__ == '__main__':
    test_jobid = sys.argv[1]

    test_results = {}
    result = test_query_job(test_jobid)
    test_results['test_query_job'] = result

    time.sleep(1)
    result = test_get_job_log(test_jobid)
    test_results['test_get_job_log'] = result

    time.sleep(1)
    result = test_job_config(test_jobid)
    test_results['test_job_config'] = result

    time.sleep(1)
    result = check_component_metric_all(test_jobid)
    test_results['check_component_metric_all'] = result

    for test_item, test_result in test_results.items():
        print("{} is success: {}".format(test_item, test_result))
