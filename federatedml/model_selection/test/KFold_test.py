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

import unittest

import numpy as np

from arch.api import eggroll
from federatedml.feature.instance import Instance
from federatedml.model_selection import KFold


class TestKFlod(unittest.TestCase):
    def setUp(self):
        eggroll.init("123")
        self.data_num = 1000
        self.feature_num = 200
        final_result = []
        for i in range(self.data_num):
            tmp = i * np.ones(self.feature_num)
            inst = Instance(inst_id=i, features=tmp, label=0)
            tmp = (str(i), inst)
            final_result.append(tmp)
        table = eggroll.parallelize(final_result,
                                    include_key=True,
                                    partition=3)
        self.table = table

    def test_split(self):
        n_splits = 10
        kfold_obj = KFold(n_splits)

        print(self.table, self.table.count())
        data_generator = kfold_obj.split(self.table)
        expect_test_data_num = self.data_num / 10
        expect_train_data_num = self.data_num - expect_test_data_num
        print("expect_train_data_num: {}, expect_test_data_num: {}".format(
            expect_train_data_num, expect_test_data_num
        ))
        for train_data, test_data in data_generator:
            train_num = train_data.count()
            test_num = test_data.count()
            print("train_num: {}, test_num: {}".format(train_num, test_num))
            self.assertTrue(0.9 * expect_train_data_num < train_num < 1.1 * expect_train_data_num)
            self.assertTrue(0.9 * expect_test_data_num < test_num < 1.1 * expect_test_data_num)


if __name__ == '__main__':
    unittest.main()
