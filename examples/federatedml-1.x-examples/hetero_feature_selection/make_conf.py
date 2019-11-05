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
import os

home_dir = os.path.split(os.path.realpath(__file__))[0]

DATA_SET = 'default_credit'
MODE = "hetero"


def get_file_path():
    guest_file_name = '_'.join([MODE, DATA_SET, 'train', 'guest'])
    host_file_name = '_'.join([MODE, DATA_SET, 'train', 'host'])
    test_guest_file_name = '_'.join([MODE, DATA_SET, 'test', 'guest'])
    test_host_file_name = '_'.join([MODE, DATA_SET, 'test', 'host'])
    return guest_file_name, host_file_name, test_guest_file_name, test_host_file_name


def write_upload_conf(json_info, guest_file_name, role):
    file_path = home_dir + '/data/' + guest_file_name + '.csv'
    config_file_path = home_dir + '/user_config/' + role + '.json'
    name_space = '_'.join([MODE, DATA_SET])

    json_info['file'] = file_path
    json_info['table_name'] = guest_file_name
    json_info['namespace'] = name_space

    config = json.dumps(json_info, indent=4)
    with open(config_file_path, "w") as fout:
        # print("path:{}".format(config_path))
        fout.write(config + "\n")


def make_upload_conf():
    original_upload_conf = home_dir + '/upload_data_guest.json'
    with open(original_upload_conf, 'r', encoding='utf-8') as f:
        json_info = json.loads(f.read())

    guest_file_name, host_file_name, test_guest_file_name, test_host_file_name = get_file_path()
    write_upload_conf(json_info, guest_file_name, 'train_guest')
    write_upload_conf(json_info, host_file_name, 'train_host')



if __name__ == '__main__':
    make_upload_conf()