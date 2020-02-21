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
import sys
import traceback

import pandas as pd

from examples.running_tools import run_config
from examples.running_tools.base_task import BaseTask

# valid data set: "breast", "default_credit", "give_credit", "vehicle"
file_path1 = '/data/projects/fate/python_mc/examples/data/dc.csv'
file_path2 = '/data/projects/fate/python_mc/examples/data/training_data_phone.csv'

f1_header = None
f2_header = 0

output_file_name = 'dc_data.csv'


class MergeData(BaseTask):
    def merge_hetero(self):
        df1 = pd.read_csv(file_path1, index_col=0, header=f1_header)
        print('df1', df1)
        df2 = pd.read_csv(file_path2, index_col=0, header=f2_header)
        print('df2', df2)
        total_train_df = pd.concat([df2, df1], axis=0, sort=False, join='inner')
        total_train_df.to_csv(run_config.TEMP_DATA_PATH + output_file_name)


if __name__ == '__main__':
    merge_data = MergeData()
    merge_data.merge_hetero()
