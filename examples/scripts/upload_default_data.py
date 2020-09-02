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
import time

from examples.test import submit


def check_data_count(submitter, fate_home, table_name, namespace, expect_count):
    fate_flow_path = os.path.join(fate_home, "../fate_flow/fate_flow_client.py")
    cmd = ["python", fate_flow_path, "-f", "table_info", "-t", table_name, "-n", namespace]
    stdout = submitter.run_cmd(cmd)
    try:
        stdout = json.loads(stdout)
        count = stdout["data"]["count"]
        if count != expect_count:
            raise AssertionError("Count of upload file is not as expect, count is: {},"
                                 "expect is: {}".format(count, expect_count))
    except:
        raise RuntimeError(f"check data error, stdout: {stdout}")

    print(f"[{time.strftime('%Y-%m-%d %X')}] check_data_out {stdout} \n")


def data_upload(submitter, upload_config, check_interval, fate_home):
    # with open(file_name) as f:
    #     upload_config = json.loads(f.read())

    task_data = upload_config["data"]
    for data in task_data:
        format_msg = f"@{data['file']} >> {data['namespace']}.{data['table_name']}"
        print(f"[{time.strftime('%Y-%m-%d %X')}]uploading {format_msg}")
        job_id = submitter.upload(data_path=data["file"],
                                  namespace=data["namespace"],
                                  name=data["table_name"],
                                  partition=data["partition"],
                                  head=data["head"])
        print(f"[{time.strftime('%Y-%m-%d %X')}]upload done {format_msg}, job_id={job_id}\n")

        submitter.await_finish(job_id, check_interval=check_interval)
        check_data_count(submitter, fate_home, data["table_name"], data["namespace"], data["count"])


def read_data(fate_home, config_type):
    if config_type == 'min-test':
        config_file = os.path.join(fate_home, "scripts/min_test_config.json")
    else:
        config_file = os.path.join(fate_home, "scripts/config.json")

    with open(config_file, 'r', encoding='utf-8') as f:
        json_info = json.loads(f.read())
    return json_info


def main():
    import examples
    fate_home = os.path.dirname(examples.__file__)
    # fate_home = os.path.abspath(f"{os.getcwd()}/../")

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-m", "--mode", type=int, help="work mode", choices=[0, 1],
                            required=True)
    arg_parser.add_argument("-f", "--force",
                            help="table existing strategy, "
                                 "-1 means skip upload, "
                                 "0 means force upload, "
                                 "1 means upload after deleting old table",
                            type=int,
                            choices=[-1, 0, 1],
                            default=0)
    arg_parser.add_argument("-i", "--interval", type=int, help="check job status every i seconds, defaults to 1",
                            default=1)

    arg_parser.add_argument("-b", "--backend", type=int, help="backend", choices=[0, 1], default=0)
    arg_parser.add_argument("-c", "--config_file", type=str, help="config file",
                            choices=["all", "min-test"], default="min-test")

    args = arg_parser.parse_args()

    work_mode = args.mode
    existing_strategy = args.force
    backend = args.backend
    interval = args.interval
    config_file = args.config_file
    spark_submit_config = {}
    submitter = submit.Submitter(fate_home=fate_home,
                                 work_mode=work_mode,
                                 backend=backend,
                                 existing_strategy=existing_strategy,
                                 spark_submit_config=spark_submit_config)

    upload_data = read_data(fate_home, config_file)

    data_upload(submitter, upload_data, interval, fate_home)


if __name__ == "__main__":
    main()
