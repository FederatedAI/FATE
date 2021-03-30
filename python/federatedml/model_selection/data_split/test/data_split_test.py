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
import uuid

import numpy as np

from fate_arch.common import profile
from fate_arch.session import Session
from federatedml.feature.instance import Instance
from federatedml.model_selection.data_split import data_split
from federatedml.param.data_split_param import DataSplitParam

profile._PROFILE_LOG_ENABLED = False


class TestDataSplit(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        self.session = Session.create(0, 0).init_computing(self.job_id).computing
        self.data_splitter = data_split.DataSplitter()
        param_dict = {"random_state": 42,
                  "test_size": 0.2, "train_size": 0.6, "validate_size": 0.2,
                  "stratified": True, "shuffle": True, "split_points": [0.5, 0.2]}
        params = DataSplitParam(**param_dict)
        self.data_splitter._init_model(params)

    def prepare_data(self, data_num, feature_num):
        final_result = []
        for i in range(data_num):
            tmp = i * np.ones(feature_num)
            label_tmp = np.random.random(1)[0]
            inst = Instance(inst_id=i, features=tmp, label=label_tmp)
            tmp = (i, inst)
            final_result.append(tmp)
        table = self.session.parallelize(final_result,
                                    include_key=True,
                                    partition=3)
        return table

    def test_transform_regression_label(self, data_num=100):
        data_instances = self.prepare_data(data_num, feature_num=10)

        expect_class_count = len(self.data_splitter.split_points) + 1

        bin_y = self.data_splitter.transform_regression_label(data_instances)
        bin_class_count = len(set(bin_y))

        self.assertEqual(expect_class_count, bin_class_count)

    def test_get_train_test_size(self):
        expect_validate_size, expect_test_size = 0.5, 0.5
        validate_size, test_size = self.data_splitter.get_train_test_size(self.data_splitter.test_size,
                                                                          self.data_splitter.validate_size)

        self.assertAlmostEqual(expect_test_size, test_size)
        self.assertAlmostEqual(expect_validate_size, validate_size)

    def test_get_class_freq(self):
        y = [1] * 10 + [0] * 3 + [1] * 20 + [2] * 10 + [0] * 2 + [2] * 5
        expect_freq_0 = 5
        expect_freq_1 = 30
        expect_freq_2 = 15

        freq_dict = data_split.DataSplitter.get_class_freq(y, label_names = [0, 1, 2])

        self.assertAlmostEqual(expect_freq_0, freq_dict[0])
        self.assertAlmostEqual(expect_freq_1, freq_dict[1])
        self.assertAlmostEqual(expect_freq_2, freq_dict[2])

    def tearDown(self):
        self.session.stop()
        try:
            self.session.cleanup("*", self.job_id)
        except EnvironmentError:
            pass


if __name__ == '__main__':
    unittest.main()
