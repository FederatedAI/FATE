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
import math
import sys
import time
import traceback

import pandas as pd

from examples.running_tools import run_config
from examples.running_tools.base_task import BaseTask

# valid data set: "breast", "default_credit", "give_credit", "vehicle"
DATA_SET = 'HomeCredit_homo_guest8'
MODE = "hetero"
PARTITION_NUM = 10

ROLE = 'guest'

TRAIN_RATIO = 0.8  # Ratio of training set, test set will be 1 - TRAIN_RATIO
GUEST_RATIO = 0.5  # For hetero, means feature number ratio, for homo means sample ratio.
HOST_RATIOS = [0.2, 0.3]

need_generate_data = True
split_train_test_only = True

add_empty_column = False


class DataTask(BaseTask):
    @staticmethod
    def get_file_names():
        train_file_names = ['_'.join([MODE, DATA_SET, 'train', ROLE])]
        test_file_names = ['_'.join([MODE, DATA_SET, 'test', ROLE])]

        if not split_train_test_only:
            for idx, _ in enumerate(HOST_RATIOS):
                train_file_names.append('_'.join([MODE, DATA_SET, 'train', 'host', str(idx)]))
                test_file_names.append('_'.join([MODE, DATA_SET, 'test', 'host', str(idx)]))

        return train_file_names, test_file_names

    @staticmethod
    def __write_upload_conf(json_info, file_name):
        partition_num = str(PARTITION_NUM) + 'p'

        file_path = run_config.TEMP_DATA_PATH + file_name + '.csv'
        config_file_path = run_config.TEMP_CONFIG_PATH + file_name + '.json'
        name_space = '_'.join([MODE, DATA_SET])

        json_info['file'] = file_path
        json_info['table_name'] = '_'.join([file_name, partition_num])
        json_info['namespace'] = name_space
        json_info['work_mode'] = run_config.WORK_MODE
        json_info['partition'] = PARTITION_NUM

        config = json.dumps(json_info, indent=4)
        with open(config_file_path, "w") as fout:
            fout.write(config + "\n")
        return config_file_path

    def make_upload_conf(self):
        original_upload_conf = run_config.UPLOAD_TEMPLATE
        with open(original_upload_conf, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())

        train_file_names, test_file_names = self.get_file_names()
        config_paths = [self.__write_upload_conf(json_info, x) for x in train_file_names + test_file_names]

        return config_paths

    def generate_data_files(self):
        data_path = run_config.DATA_PATH + DATA_SET + '.csv'
        total_data_df = pd.read_csv(data_path, index_col=0)
        if add_empty_column:
            total_data_df['feature_0'] = 0.00001
        train_file_names, test_file_names = self.get_file_names()
        train_data, test_data = self.__split_train_test(total_data_df)

        if not split_train_test_only:
            if MODE == 'hetero':
                self.__split_hetero(train_data, train_file_names)
                self.__split_hetero(test_data, test_file_names)
            else:
                self.__split_homo(train_data, train_file_names)
                self.__split_homo(test_data, test_file_names)
        else:
            train_data.to_csv(run_config.TEMP_DATA_PATH + train_file_names[0] + '.csv')
            test_data.to_csv(run_config.TEMP_DATA_PATH + test_file_names[0] + '.csv')


    @staticmethod
    def __split_homo(total_train_df, file_names):
        guest_num = math.floor(total_train_df.shape[0] * GUEST_RATIO)
        guest_df = total_train_df[0: guest_num]
        guest_df.to_csv(run_config.TEMP_DATA_PATH + file_names[0] + '.csv')

        last_num = guest_num
        for idx, host_ratio in enumerate(HOST_RATIOS[: -1]):
            host_num = math.floor(total_train_df.shape[0] * host_ratio)
            host_df = total_train_df[last_num: last_num + host_num]
            host_df.to_csv(run_config.TEMP_DATA_PATH + file_names[idx + 1] + '.csv')
            last_num += host_num
        last_host_df = total_train_df.iloc[last_num:]
        last_host_df.to_csv(run_config.TEMP_DATA_PATH + file_names[-1] + '.csv')

    @staticmethod
    def __split_hetero(total_train_df, file_names):
        feature_nums = total_train_df.shape[1]
        guest_feature_nums = math.floor(feature_nums * GUEST_RATIO)

        guest_df = total_train_df.iloc[:, :guest_feature_nums]
        guest_df.to_csv(run_config.TEMP_DATA_PATH + file_names[0] + '.csv')

        last_num = guest_feature_nums
        for idx, host_ratio in enumerate(HOST_RATIOS[: -1]):
            host_feature_nums = math.floor(feature_nums * host_ratio)
            host_df = total_train_df.iloc[:, last_num: last_num + host_feature_nums]
            host_df.to_csv(run_config.TEMP_DATA_PATH + file_names[idx + 1] + '.csv')
            last_num += host_feature_nums
        last_host_df = total_train_df.iloc[:, last_num:]
        last_host_df.to_csv(run_config.TEMP_DATA_PATH + file_names[-1] + '.csv')

    @staticmethod
    def __split_train_test(total_train_df):
        total_num = total_train_df.shape[0]
        train_num = int(total_num * TRAIN_RATIO)
        train_data = total_train_df.iloc[:train_num]
        test_data = total_train_df.iloc[train_num:]
        return train_data, test_data

    @staticmethod
    def save_loaded_info(uploaded_files):
        saved_result_path = run_config.SAVE_RESULT_PATH
        with open(saved_result_path, 'r', encoding='utf-8') as f:
            saved_result_json = json.loads(f.read())
        uploaded_table_names = saved_result_json.get('uploaded_table_names')
        uploaded_namespaces = saved_result_json.get('uploaded_namespaces')
        for table_name, namespace in uploaded_files:
            uploaded_table_names.append(table_name)
            uploaded_namespaces.append(namespace)
        saved_result_json['uploaded_table_names'] = uploaded_table_names
        saved_result_json['uploaded_namespaces'] = uploaded_namespaces
        config = json.dumps(saved_result_json, indent=4)
        with open(saved_result_path, "w") as fout:
            fout.write(config + "\n")

    def check_tables(self, uploaded_files=None):
        if uploaded_files is not None:
            for table_name, namespace in uploaded_files:
                self.get_table_info(table_name, namespace)
        else:
            saved_result_path = run_config.SAVE_RESULT_PATH
            with open(saved_result_path, 'r', encoding='utf-8') as f:
                saved_result_json = json.loads(f.read())
            uploaded_table_names = saved_result_json.get('uploaded_table_names')
            uploaded_namespaces = saved_result_json.get('uploaded_namespaces')
            for table_name, namespace in zip(uploaded_table_names, uploaded_namespaces):
                self.get_table_info(table_name, namespace)

    def upload_data(self):
        if need_generate_data:
            self.generate_data_files()
        config_paths = self.make_upload_conf()
        uploaded_files = []
        for path in config_paths:
            cmd = ['python', run_config.FATE_FLOW_PATH, "-f", "upload", "-c", path]
            stdout = self.start_task(cmd)
            try:
                table_info = (stdout['data']['table_name'], stdout['data']['namespace'])
                uploaded_files.append(table_info)
            except ValueError:
                raise ValueError("Cannot obtain table info from stdout, stdout is : {}".format(stdout))
            time.sleep(3)
        self.save_loaded_info(uploaded_files)
        self.check_tables(uploaded_files)

    def destroy_data(self):
        saved_result_path = run_config.SAVE_RESULT_PATH
        with open(saved_result_path, 'r', encoding='utf-8') as f:
            saved_result_json = json.loads(f.read())
        uploaded_table_names = saved_result_json.get('uploaded_table_names')
        uploaded_namespaces = saved_result_json.get('uploaded_namespaces')
        for table_name, namespace in zip(uploaded_table_names, uploaded_namespaces):
            cmd = ["python", run_config.FATE_FLOW_PATH, "-f", "table_delete", "-t", table_name, "-n", namespace]
            stdout = self.start_task(cmd)
            print("deleting {}, {}, stdout: {}".format(table_name, namespace, stdout))
        saved_result_json['uploaded_table_names'] = []
        saved_result_json['uploaded_namespaces'] = []
        config = json.dumps(saved_result_json, indent=4)
        with open(saved_result_path, "w") as fout:
            fout.write(config + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--func', required=False, type=str, help="run function",
                        choices=('upload', 'check', 'destroy'), default='upload'
                        )

    try:
        args = parser.parse_args()
        data_task_obj = DataTask()
        if args.func == 'upload':
            data_task_obj.upload_data()
        elif args.func == 'check':
            data_task_obj.check_tables()
        elif args.func == 'destroy':
            data_task_obj.destroy_data()

    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        response = {'retcode': 100, 'retmsg': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
        print(json.dumps(response, indent=4))
        print()
