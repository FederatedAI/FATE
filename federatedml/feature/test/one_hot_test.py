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


import unittest

from arch.api import session

session.init("123")
from federatedml.feature.one_hot_encoder import OneHotEncoder
from federatedml.feature.instance import Instance
import numpy as np


class TestOneHotEncoder(unittest.TestCase):
    def setUp(self):
        self.data_num = 100
        self.feature_num = 3
        self.cols = [0, 1, 2]
        self.header = ['x' + str(i) for i in range(self.feature_num)]
        final_result = []

        for i in range(self.data_num):
            tmp = []
            for _ in range(self.feature_num):
                tmp.append(np.random.choice([1, 2, 3]))
            tmp = np.array(tmp)
            inst = Instance(inst_id=i, features=tmp, label=0)
            tmp_pair = (str(i), inst)
            final_result.append(tmp_pair)

        table = session.parallelize(final_result,
                                    include_key=True,
                                    partition=10)
        table.schema = {"header": self.header}
        self.model_name = 'OneHotEncoder'

        self.table = table

        self.args = {"data": {self.model_name: {"data": table}}}

    def test_instance(self):
        one_hot_encoder = OneHotEncoder()
        one_hot_encoder.cols = self.cols
        one_hot_encoder.cols_index = self.cols

        result = one_hot_encoder.fit(self.table)
        local_result = result.collect()
        for k, v in local_result:
            new_features = v.features
            self.assertTrue(len(new_features) == self.feature_num * 3)


if __name__ == '__main__':
    unittest.main()
