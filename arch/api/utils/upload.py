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
#

import json
import argparse
import os
import traceback
import csv
import sys
import time
from arch.api import eggroll, storage

CSV = 'csv'
LOAD_DATA_COUNT = 10000
MAX_PARTITION_NUM = 1024


def list_to_str(input_list):
    str1 = ''
    size = len(input_list)
    for i in range(size):
        if i == size - 1:
            str1 += str(input_list[i])
        else:
            str1 += str(input_list[i]) + ','

    return str1


def save_data_header(header_source, dst_table_name, dst_table_namespace):
    header_source_item = header_source.split(',')
    storage.save_data_table_meta({'header': ','.join(header_source_item[1:]).strip(), 'sid': header_source_item[0]}, dst_table_name,
                                 dst_table_namespace)


def read_data(input_file, dst_table_name, dst_table_namespace, head=True):
    split_file_name = input_file.split('.')
    if CSV in split_file_name:
        with open(input_file) as csv_file:
            csv_reader = csv.reader(csv_file)
            if head is True:
                data_head = next(csv_reader)
                save_data_header(','.join(data_head), dst_table_name, dst_table_namespace)

            for row in csv_reader:
                yield (row[0], list_to_str(row[1:]))
    else:
        with open(input_file, 'r') as fin:
            if head is True:
                data_head = fin.readline()
                save_data_header(data_head, dst_table_name, dst_table_namespace)

            lines = fin.readlines()
            for line in lines:
                values = line.replace("\n", "").replace("\t", ",").split(",")
                yield (values[0], list_to_str(values[1:]))


def generate_table_name(input_file_path):
    local_time = time.localtime(time.time())
    str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    file_name = input_file_path.split(".")[0]
    file_name = file_name.split("/")[-1]
    return file_name, str_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_id', required=True, type=str, help="job id to use")
    parser.add_argument('-c', '--config', required=True, type=str,
                        help="you should provide a path of configure file with json format")
    try:
        args = parser.parse_args()
        job_config = {}
        try:
            args.config = os.path.abspath(args.config)
            input_file_path = None
            head = True
            table_name = None
            namespace = None
            with open(args.config, 'r') as f:
                job_config = json.load(f)

                try:
                    input_file_path = job_config['file']
                except:
                    traceback.print_exc()

                try:
                    read_head = job_config['head']
                    if read_head == 0:
                        head = False
                    elif read_head == 1:
                        head = True
                except:
                    print("'head' in .json should be 0 or 1, set head to 1")

                try:
                    partition = job_config['partition']
                    if partition <= 0 or partition > MAX_PARTITION_NUM:
                        print("Error number of partition, it should between %d and %d" % (0, MAX_PARTITION_NUM))
                        sys.exit()
                except:
                    print("set partition to 1")
                    partition = 1

                try:
                    table_name = job_config['table_name']
                except:
                    print("not setting table_name or setting error, set table_name according to current time")

                try:
                    namespace = job_config['namespace']
                except:
                    print("not setting namespace or setting error, set namespace according to input file name")

                work_mode = job_config.get('work_mode')
                if work_mode is None:
                    work_mode = 0

            if not os.path.exists(input_file_path):
                print("%s is not exist, please check the configure" % (input_file_path))
                sys.exit()

            _namespace, _table_name = generate_table_name(input_file_path)
            if namespace is None:
                namespace = _namespace
            if table_name is None:
                table_name = _table_name
            eggroll.init(job_id=args.job_id, mode=work_mode)
            input_data = read_data(input_file_path, table_name, namespace, head)
            in_version = job_config.get('in_version', False)
            data_table = storage.save_data(input_data, name=table_name, namespace=namespace, partition=partition, in_version=in_version)
            print("------------load data finish!-----------------")
            print("file: {}".format(input_file_path))
            print("total data_count: {}".format(data_table.count()))
            print("table name: {}, table namespace: {}".format(table_name, namespace))

        except ValueError:
            print('json parse error')
            exit(-102)
        except IOError:
            print('read file error')
            exit(-103)
    except:
        traceback.print_exc()
