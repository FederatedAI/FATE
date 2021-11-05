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
import sys
import time

cur_path = os.path.realpath(__file__)
for i in range(3):
    cur_path = os.path.dirname(cur_path)
print(f'fate_path: {cur_path}')
sys.path.append(cur_path)

from examples.scripts import submit
from python.fate_client.flow_sdk.client import FlowClient


def check_data_count(submitter, table_name, namespace, expect_count):
    command = "table/info"
    config_data = {
        "table_name": table_name,
        "namespace": namespace
    }
    stdout = submitter.run_flow_client(command=command, config_data=config_data)
    try:
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
        if job_id is None:
            print("table already exist. To upload again, Please add '-f 1' in start cmd")
            continue

        submitter.await_finish(job_id, check_interval=check_interval)
        check_data_count(submitter, data["table_name"], data["namespace"], data["count"])


def read_data(fate_home, config_type):
    if config_type == 'min-test':
        config_file = os.path.join(fate_home, "examples/scripts/min_test_config.json")
    else:
        config_file = os.path.join(fate_home, "examples/scripts/config.json")

    with open(config_file, 'r', encoding='utf-8') as f:
        json_info = json.loads(f.read())
    return json_info


def main():
    # import examples
    # fate_home = os.path.dirname(examples.__file__)
    # fate_home = os.path.abspath(f"{os.getcwd()}/../")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-f", "--force",
                            help="table existing strategy, "
                                 "0 means force upload, "
                                 "1 means upload after deleting old table",
                            type=int,
                            choices=[0, 1],
                            default=0)
    arg_parser.add_argument("-i", "--interval", type=int, help="check job status every i seconds, defaults to 1",
                            default=1)

    arg_parser.add_argument("-ip", "--flow_server_ip", type=str, help="please input flow server'ip")
    arg_parser.add_argument("-port", "--flow_server_port", type=int, help="please input flow server port")
    arg_parser.add_argument("-c", "--config_file", type=str, help="config file", default="min-test")

    args = arg_parser.parse_args()

    existing_strategy = args.force
    interval = args.interval
    config_file = args.config_file
    ip = args.flow_server_ip
    port = args.flow_server_port
    flow_client = FlowClient(ip=ip, port=port, version="v1")

    spark_submit_config = {}
    submitter = submit.Submitter(flow_client=flow_client,
                                 fate_home=cur_path,
                                 existing_strategy=existing_strategy,
                                 spark_submit_config=spark_submit_config)

    if config_file in ["all", "min-test"]:
        upload_data = read_data(cur_path, config_file)
    else:
        with open(config_file, 'r', encoding='utf-8') as f:
            upload_data = json.loads(f.read())

    data_upload(submitter, upload_data, interval, cur_path)


if __name__ == "__main__":
    main()
