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
import subprocess
import sys
import time
import traceback
import os

feature_idx = -1

WORK_MODE = 0

MAX_INTERSECT_TIME = 600
MAX_TRAIN_TIME = 3600
OTHER_TASK_TIME = 300
STATUS_CHECKER_TIME = 10
home_dir = os.path.split(os.path.realpath(__file__))[0]
fate_flow_path = home_dir + "/../../python/fate_flow/fate_flow_client.py"
fate_flow_home = home_dir + "/../../python/fate_flow"


class BaseTask(object):
    def __init__(self, argv=None):
        if argv is not None:
            self._parse_argv(argv)

    def _parse_argv(self, argv):
        pass

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
        except json.JSONDecodeError:
            raise RuntimeError("start task error, return value: {}".format(stdout))
        return stdout

    def get_table_info(self, name, namespace):
        # cmd = ['flow', "table", "info", "-t", str(name), "-n", str(namespace)]
        cmd = ['python', fate_flow_path, "-f", "table_info", "-t", str(name), "-n", str(namespace)]
        table_info = self.start_task(cmd)
        return table_info

    def read_json_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            result_json = json.loads(f.read())
        return result_json

    def write_json_file(self, json_info, file_path):
        config = json.dumps(json_info, indent=4)
        with open(file_path, "w") as fout:
            fout.write(config + "\n")


class QuerySchema(BaseTask):
    def __init__(self, args):
        self.role = args.role
        self.party_id = args.party_id

    def query_component_output_data(self, job_id, cpn, this_feature_idx=None):
        if this_feature_idx is None:
            this_feature_idx = feature_idx
        cmd = ['python', fate_flow_path, "-f", "component_output_data_table", "-j", job_id,
               '-cpn', cpn, '-r', self.role, '-p', str(self.party_id)]
        # cmd = ['flow', 'component', 'output-data-table', '-j', job_id, '-cpn', cpn, "-r", self.role,
        #        '-p', self.party_id]

        stdout = self.start_task(cmd)
        if stdout['retcode'] != 0:
            raise RuntimeError(f"checking component output data error, error info: {stdout['retmsg']}")

        print(f"stdout: {stdout}")
        table_name = stdout.get('data')[0]['table_name']
        table_namespace = stdout.get('data')[0]['table_namespace']
        table_info = self.get_table_info(table_name, table_namespace)
        print("query_component_output_data result: {}".format(table_info))
        try:
            header = table_info['data']['schema']['header']
        except ValueError as e:
            raise ValueError(f"Obtain header from table error, error msg: {e}")

        result = []
        for idx, header_name in enumerate(header[1:]):
            if this_feature_idx == -1 or idx in this_feature_idx:
                result.append((idx, header_name))
        print("Queried header is {}".format(result))
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('-c', '--conf', required=False, type=str, help="input conf path", default=None)
    parser.add_argument('-cpn', '--component_name', required=False, type=str, help="component name",
                        default='dataio_0')
    parser.add_argument('-j', '--job_id', required=True, type=str, help="job id")
    parser.add_argument('-r', '--role', required=True, choices=["guest", "host", "arbiter"],
                        type=str, help="job id")
    parser.add_argument('-p', '--party_id', required=True, type=str, help="party id")

    try:
        args = parser.parse_args()
        task_obj = QuerySchema(args)
        task_obj.query_component_output_data(str(args.job_id), args.component_name)

    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        response = {'retcode': 100, 'retmsg': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
        print(json.dumps(response, indent=4))
        print()
