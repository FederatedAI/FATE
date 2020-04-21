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

import numpy as np

from arch.api import session

session.init("123")

from federatedml.feature.instance import Instance
from federatedml.statistic.statics import MultivariateStatisticalSummary


class TestBaseBinningFunctions(unittest.TestCase):
    def setUp(self):
        self.table_list = []

    def _gen_data(self, label_histogram: dict, partition=10):
        label_list = []
        data_num = 0
        for y, n in label_histogram.items():
            data_num += n
            label_list.extend([y] * n)

        np.random.shuffle(label_list)
        data_insts = []

        for i in range(data_num):
            features = np.random.randn(10)
            inst = Instance(features=features, label=label_list[i])
            data_insts.append((i, inst))
        result = session.parallelize(data_insts, include_key=True, partition=partition)
        result.schema = {'header': ['d' + str(x) for x in range(10)]}
        self.table_list.append(result)
        return result

    def test_histogram(self):
        histograms = [
            {0: 100, 1: 100},
            {0: 9700, 1: 300},
            {0: 2000, 1: 18000},
            {0: 8000, 1: 2000}
        ]

        partitions = [10, 1, 48, 32]

        for i, h in enumerate(histograms):
            data = self._gen_data(h, partitions[i])
            summary_obj = MultivariateStatisticalSummary(data_instances=data)
            label_hist = summary_obj.get_label_histogram()
            self.assertDictEqual(h, label_hist)

    def tearDown(self):
        for table in self.table_list:
            table.destroy()


if __name__ == '__main__':
    unittest.main()
