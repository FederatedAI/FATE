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

_home_dir = os.path.split(os.path.realpath(__file__))[0]

CONF_PATH = _home_dir + '/../federatedml-1.x-examples/'
DATA_PATH = _home_dir + '/../data/'
TEMP_DATA_PATH = _home_dir + '/temp_data/'
TEMP_CONFIG_PATH = _home_dir + '/temp_config/'

FATE_FLOW_PATH = _home_dir + "/../../fate_flow/fate_flow_client.py"

UPLOAD_TEMPLATE = _home_dir + '/upload_data.json'
SAVE_RESULT_PATH = TEMP_CONFIG_PATH + 'saved_result.json'

WORK_MODE = 0

MAX_INTERSECT_TIME = 600
MAX_TRAIN_TIME = 3600
OTHER_TASK_TIME = 300

STATUS_CHECKER_TIME = 10

breast = 'breast.csv'
default_credit = 'default_credit.csv'

ALL_MODULES = ['DataIO', 'Evaluation', "FeatureScale", 'FederatedSample', 'HeteroFeatureBinning',
               "HeteroFeatureSelection", 'HeteroLinR', 'HeteroLR', 'HeteroPoisson', 'HeteroSecureBoost',
               'HomoLR', 'HomoNN', 'Intersection', 'LocalBaseline', 'OneHotEncoder', 'Union']

simple_hetero_lr = {"moudules": ['dataio', 'intersect', 'hetero_lr'],
                    "has_input_model": [False, False, False],
                    "has_isometric_model": [False, False, False],
                    "has_input_data": [True, True, False],
                    "has_input_train_data": [False, False, True],
                    "has_output_data": [True, True, True],
                    "has_out_model": [False, False, False]}
