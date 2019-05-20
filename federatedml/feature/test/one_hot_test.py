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

from arch.api import eggroll

eggroll.init("123")
from federatedml.feature.one_hot_encoder import OneHotEncoder
from federatedml.feature.instance import Instance
from federatedml.param.param import OneHotEncoderParam
import numpy as np


class TestOneHotEncoder(unittest.TestCase):
    def setUp(self):
        self.data_num = 10
        self.feature_num = 3
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

        table = eggroll.parallelize(final_result,
                                    include_key=True,
                                    partition=10)
        table.schema = {"header": self.header}

        self.table = table
        self.cols = ['x0']

    def test_instance(self):
        param = OneHotEncoderParam(cols=self.cols)
        one_hot_encoder = OneHotEncoder(param=param)

        one_hot_encoder.fit(self.table)
        local_data = self.table.collect()
        print("original data:")
        for k, v in local_data:
            print(k, v.features)
        new_data = one_hot_encoder.transform(data_instances=self.table)
        local_data = new_data.collect()
        print("One-hot encoded data:")
        for k, v in local_data:
            print(k, v.features)


if __name__ == '__main__':
    unittest.main()
