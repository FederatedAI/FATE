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

import json
import subprocess
import time

from examples.running_tools import run_config


class BaseTask(object):
    def __init__(self, argv=None):
        if argv is not None:
            self._parse_argv(argv)

    def _parse_argv(self, argv):
        pass

    @staticmethod
    def start_block_task(cmd, max_waiting_time=run_config.OTHER_TASK_TIME):
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
                time.sleep(run_config.STATUS_CHECKER_TIME)
            else:
                break
        stdout = json.loads(stdout)
        return stdout

    @staticmethod
    def start_block_func(run_func, params, exit_func, max_waiting_time=run_config.OTHER_TASK_TIME):
        start_time = time.time()
        while True:
            result = run_func(*params)
            if exit_func(result):
                return result
            end_time = time.time()
            if end_time - start_time >= max_waiting_time:
                return None
            time.sleep(run_config.STATUS_CHECKER_TIME)

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
        cmd = ["python", run_config.FATE_FLOW_PATH, "-f", "table_info", "-t", str(name), "-n", str(namespace)]
        table_info = self.start_task(cmd)
        print(table_info)
        return table_info

    def read_json_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            result_json = json.loads(f.read())
        return result_json

    def write_json_file(self, json_info, file_path):
        config = json.dumps(json_info, indent=4)
        with open(file_path, "w") as fout:
            fout.write(config + "\n")
