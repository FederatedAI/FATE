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

from arch.api import session
from federatedml.feature.instance import Instance
from federatedml.model_selection import KFold
from federatedml.param.cross_validation_param import CrossValidationParam


class TestKFlod(unittest.TestCase):
    def setUp(self):
        session.init("123")
        self.data_num = 1000
        self.feature_num = 200
        final_result = []
        for i in range(self.data_num):
            tmp = i * np.ones(self.feature_num)
            inst = Instance(inst_id=i, features=tmp, label=0)
            tmp = (str(i), inst)
            final_result.append(tmp)
        table = session.parallelize(final_result,
                                    include_key=True,
                                    partition=3)
        self.table = table

    def test_split(self):
        kfold_obj = KFold()
        kfold_obj.n_splits = 10
        kfold_obj.random_seed = 32

        # print(self.table, self.table.count())
        data_generator = kfold_obj.split(self.table)
        expect_test_data_num = self.data_num / 10
        expect_train_data_num = self.data_num - expect_test_data_num

        key_list = []
        for train_data, test_data in data_generator:
            train_num = train_data.count()
            test_num = test_data.count()
            # print("train_num: {}, test_num: {}".format(train_num, test_num))
            self.assertTrue(0.9 * expect_train_data_num < train_num < 1.1 * expect_train_data_num)
            self.assertTrue(0.9 * expect_test_data_num < test_num < 1.1 * expect_test_data_num)
            first_key = train_data.first()[0]
            key_list.append(first_key)

        # Test random seed work
        kfold_obj2 = KFold()
        kfold_obj2.n_splits = 10
        kfold_obj2.random_seed = 32

        data_generator = kfold_obj.split(self.table)
        n = 0
        for train_data, test_data in data_generator:
            second_key = train_data.first()[0]
            first_key = key_list[n]
            self.assertTrue(first_key == second_key)
            n += 1


if __name__ == '__main__':
    unittest.main()
