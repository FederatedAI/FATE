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

import csv
import json
import os
import subprocess
import sys
import time
import traceback

from arch.api import eggroll
from arch.api.io.feature import save_feature_data

home_dir = os.path.split(os.path.realpath(__file__))[0]
config_path = home_dir + '/conf'
data_path = home_dir + '/../data'
load_file_program = home_dir + '/../load_file/load_file.py'

# data_set = 'breast'
data_set = 'default_credit'
# data_set = 'give_credit'

CSV = 'csv'
LOAD_DATA_COUNT = 10000
MAX_PARTITION_NUM = 32


def make_config_file(work_mode, job_id, guest_partyid, host_partyid, result_table, result_namespace, scene_id, method):
    with open(config_path + '/guest_runtime_conf.json', 'r', encoding='utf-8') as load_f:
        guest_config = json.load(load_f)

    guest_config['local']['party_id'] = guest_partyid
    guest_config['local']['scene_id'] = scene_id

    guest_config['role']['host'][0] = host_partyid
    guest_config['role']['guest'][0] = guest_partyid
    guest_config['WorkFlowParam']['work_mode'] = int(work_mode)
    guest_config['FeatureBinningParam']['result_table'] = result_table
    guest_config['FeatureBinningParam']['result_namespace'] = result_namespace
    guest_config['FeatureSelectionParam']['method'] = method
    guest_config['WorkFlowParam']['train_input_table'] = data_set + '_guest_' + job_id

    guest_config_path = config_path + '/guest_runtime_conf.json_' + str(job_id)

    with open(guest_config_path, 'w', encoding='utf-8') as json_file:
        json.dump(guest_config, json_file, ensure_ascii=False)

    with open(config_path + '/host_runtime_conf.json', 'r', encoding='utf-8') as load_f:
        host_config = json.load(load_f)

    host_config['local']['party_id'] = host_partyid
    host_config['local']['scene_id'] = scene_id
    host_config['role']['host'][0] = host_partyid
    host_config['role']['guest'][0] = guest_partyid
    host_config['WorkFlowParam']['work_mode'] = int(work_mode)
    host_config['FeatureBinningParam']['result_table'] = result_table
    host_config['FeatureBinningParam']['result_namespace'] = result_namespace
    host_config['FeatureSelectionParam']['method'] = method
    host_config['WorkFlowParam']['train_input_table'] = data_set + '_host_' + job_id

    host_config_path = config_path + '/host_runtime_conf.json_' + str(job_id)
    with open(host_config_path, 'w', encoding='utf-8') as json_file:
        json.dump(host_config, json_file, ensure_ascii=False)

    with open(config_path + '/load_file.json', 'r', encoding='utf-8') as load_f:
        load_config = json.load(load_f)
    load_config['work_mode'] = work_mode
    load_config['file'] = data_path + '/' + data_set + '_b.csv'
    load_config['table_name'] = data_set + '_guest_' + job_id
    load_config['scene_id'] = scene_id
    load_config['role'] = 'guest'
    load_config['my_party_id'] = guest_partyid
    load_config['partner_party_id'] = host_partyid

    load_file_guest = config_path + '/load_file.json_guest_' + str(job_id)
    with open(load_file_guest, 'w', encoding='utf-8') as json_file:
        json.dump(load_config, json_file, ensure_ascii=False)

    load_config['file'] = data_path + '/' + data_set + '_a.csv'
    load_config['table_name'] = data_set + '_host_' + job_id
    load_config['scene_id'] = scene_id
    load_config['role'] = 'host'
    load_config['my_party_id'] = host_partyid
    load_config['partner_party_id'] = guest_partyid

    load_file_host = config_path + '/load_file.json_host_' + str(job_id)
    with open(load_file_host, 'w', encoding='utf-8') as json_file:
        json.dump(load_config, json_file, ensure_ascii=False)

    return guest_config_path, host_config_path, load_file_guest, load_file_host


def list_to_str(input_list):
    str1 = ''
    size = len(input_list)
    for i in range(size):
        if i == size - 1:
            str1 += str(input_list[i])
        else:
            str1 += str(input_list[i]) + ','

    return str1


def read_data(input_file='', head=True):
    split_file_name = input_file.split('.')
    if CSV in split_file_name:
        print("file type is csv")
        with open(input_file) as csv_file:
            csv_reader = csv.reader(csv_file)
            if head is True:
                csv_head = next(csv_reader)

            for row in csv_reader:
                yield (row[0], list_to_str(row[1:]))
    else:
        print("file type is not known, read it as txt")
        with open(input_file, 'r') as fin:
            if head is True:
                head = fin.readline()

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


def data_to_eggroll_table(data, namespace, table_name, partition=1, work_mode=0):
    eggroll.init(mode=work_mode)
    data_table = eggroll.table(table_name, namespace, partition=partition, create_if_missing=True, error_if_exist=False)
    data_table.put_all(data)
    data_table_count = data_table.count()
    print("------------load data finish!-----------------")
    print("total data_count:" + str(data_table.count()))
    print("namespace:%s, table_name:%s" % (namespace, table_name))
    # for kv in data_table.collect():
    #    print(kv)


def load_file(load_file_path):
    try:
        # args.config = os.path.abspath(args.config)
        input_file_path = None
        head = True
        table_name = None
        namespace = None
        with open(load_file_path, 'r') as f:
            data = json.load(f)
            try:
                input_file_path = data['file']
            except:
                traceback.print_exc()

            try:
                read_head = data['head']
                if read_head == 0:
                    head = False
                elif read_head == 1:
                    head = True
            except:
                print("'head' in .json should be 0 or 1, set head to 1")

            try:
                partition = data['partition']
                if partition <= 0 or partition > MAX_PARTITION_NUM:
                    print("Error number of partition, it should between %d and %d" % (0, MAX_PARTITION_NUM))
                    sys.exit()
            except:
                print("set partition to 1")
                partition = 1

            try:
                table_name = data['table_name']
            except:
                print("not setting table_name or setting error, set table_name according to current time")

            try:
                namespace = data['namespace']
            except:
                print("not setting namespace or setting error, set namespace according to input file name")

            work_mode = data.get('work_mode')
            if work_mode is None:
                work_mode = 0
            else:
                work_mode = int(work_mode)

        if not os.path.exists(input_file_path):
            print("%s is not exist, please check the configure" % (input_file_path))
            sys.exit()

        input_data = read_data(input_file_path, head)
        if data.get("scene_id") and data.get("role") and data.get("my_party_id") and data.get("partner_party_id"):
            eggroll.init(mode=work_mode)
            save_feature_data(input_data,
                              scene_id=data["scene_id"],
                              my_role=data["role"],
                              my_party_id=data["my_party_id"],
                              partner_party_id=data["partner_party_id"]
                              )
        else:
            _namespace, _table_name = generate_table_name(input_file_path)
            if namespace is None:
                namespace = _namespace
            if table_name is None:
                table_name = _table_name
            data_to_eggroll_table(input_data, namespace, table_name, partition, work_mode)

    except ValueError:
        print('json parse error')
        exit(-102)
    except IOError:
        print('read file error')
        exit(-103)


if __name__ == '__main__':
    work_mode = sys.argv[1]
    jobid = sys.argv[2]
    guest_partyid = int(sys.argv[3])
    host_partyid = int(sys.argv[4])
    result_table = sys.argv[5]
    result_namespace = sys.argv[6]
    scene_id = int(sys.argv[7])
    method = sys.argv[8]

    guest_config_path, host_config_path, load_file_guest, load_file_host = make_config_file(work_mode, jobid,
                                                                                            guest_partyid, host_partyid,
                                                                                            result_table,
                                                                                            result_namespace,
                                                                                            scene_id,
                                                                                            method)
    load_file(load_file_guest)
    load_file(load_file_host)

    work_path = home_dir + '/../../workflow/hetero_feature_selection_workflow/hetero_feature_selection_guest_workflow.py'
    subprocess.Popen(["python",
                      work_path,
                      "-c",
                      guest_config_path,
                      "-j",
                      jobid
                      ])

    work_path = home_dir + '/../../workflow/hetero_feature_selection_workflow/hetero_feature_selection_host_workflow.py'
    subprocess.Popen(["python",
                      work_path,
                      "-c",
                      host_config_path,
                      "-j",
                      jobid
                      ])
