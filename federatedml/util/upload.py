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
import os
import sys
import time

from arch.api import eggroll,storage

from arch.api.utils import log_utils, file_utils, dtable_utils

LOGGER = log_utils.getLogger()


class UpLoad(object):
    def __init__(self):
        self.taskid = ''
        self.tracker = None
        self.MAX_PARTITION_NUM = 1024
        self.parameters = {}

    def run(self, component_parameters=None, args=None):
        self.parameters = component_parameters["UpLoadParam"]
        self.parameters["role"] = component_parameters["role"]
        self.parameters["local"] = component_parameters["local"]
        if not os.path.isabs(self.parameters.get("file", "")):
            self.parameters["file"] = os.path.join(file_utils.get_project_base_directory(), self.parameters["file"])
        if not os.path.exists(self.parameters["file"]):
            print("%s is not exist, please check the configure" % (self.parameters["file"]))
        table_name, namespace = dtable_utils.get_table_info(config=self.parameters,
                                                            create=True)
        _namespace, _table_name = self.generate_table_name(self.parameters["file"])
        if namespace is None:
            namespace = _namespace
        if table_name is None:
            table_name = _table_name
        eggroll.init(mode=self.parameters['work_mode'])
        read_head = self.parameters['head']
        head = True
        if read_head == 0:
            head = False
        elif read_head == 1:
            head = True
        else:
            print("'head' in .json should be 0 or 1, set head to 1")
        partition = self.parameters["partition"]
        try:
            if partition <=0 or partition >= self.MAX_PARTITION_NUM :
                print("Error number of partition, it should between %d and %d" % (0, self.MAX_PARTITION_NUM))
                sys.exit()
        except:
            print("set partition to 1")
            self.parameters["partition"] = 1

        input_data = self.read_data(table_name, namespace, head)
        data_table = storage.save_data(input_data, name=table_name, namespace=namespace, partition=self.parameters["partition"])
        print("------------load data finish!-----------------")
        print("file: {}".format(self.parameters["file"]))
        print("total data_count: {}".format(data_table.count()))
        print("table name: {}, table namespace: {}".format(table_name, namespace))

    def set_taskid(self, taskid):
        self.taskid = taskid

    def set_tracker(self, tracker):
        self.tracker = tracker

    def read_data(self, dst_table_name, dst_table_namespace, head=True):
        input_file = self.parameters["file"]
        split_file_name = input_file.split('.')
        if 'csv' in split_file_name:
            with open(input_file) as csv_file:
                csv_reader = csv.reader(csv_file)
                if head is True:
                    data_head = next(csv_reader)
                    self.save_data_header(','.join(data_head), dst_table_name, dst_table_namespace)

                for row in csv_reader:
                    yield (row[0], self.list_to_str(row[1:]))
        else:
            with open(input_file, 'r') as fin:
                if head is True:
                    data_head = fin.readline()
                    self.save_data_header(data_head, dst_table_name, dst_table_namespace)

                lines = fin.readlines()
                for line in lines:
                    values = line.replace("\n", "").replace("\t", ",").split(",")
                    yield (values[0], self.list_to_str(values[1:]))

    def save_data_header(self, header_source, dst_table_name, dst_table_namespace):
        storage.save_data_table_meta({'header': ','.join(header_source.split(',')[1:]).strip()}, dst_table_name,
                                     dst_table_namespace)

    def list_to_str(self, input_list):
        str1 = ''
        size = len(input_list)
        for i in range(size):
            if i == size - 1:
                str1 += str(input_list[i])
            else:
                str1 += str(input_list[i]) + ','

        return str1

    def generate_table_name(self, input_file_path):
        local_time = time.localtime(time.time())
        str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        file_name = input_file_path.split(".")[0]
        file_name = file_name.split("/")[-1]
        return file_name, str_time