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

from arch.api import session
from federatedml.tree import FeatureHistogram
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.util import consts
import copy
import numpy as np
import random


class TestFeatureHistogram(unittest.TestCase):
    def setUp(self):
        self.feature_histogram = FeatureHistogram()
        session.init("test_feature_histogram")
        data_insts = []
        for i in range(1000):
            indices = []
            data = []
            for j in range(10):
                x = random.randint(0, 5)
                if x != 0:
                    data.append(x)
                    indices.append(j)
            sparse_vec = SparseVector(indices, data, shape=10)
            data_insts.append((Instance(features=sparse_vec), (1, random.randint(0, 3))))
        self.node_map = {0: 0, 1: 1, 2: 2, 3: 3}
        self.data_insts = data_insts
        self.data_bin = session.parallelize(data_insts, include_key=False)

        self.grad_and_hess_list = [(random.random(), random.random()) for i in range(1000)]
        self.grad_and_hess = session.parallelize(self.grad_and_hess_list, include_key=False)

        bin_split_points = []
        for i in range(10):
            bin_split_points.append(np.array([i for i in range(5)]))
        self.bin_split_points = np.array(bin_split_points)
        self.bin_sparse = [0 for i in range(10)]

    def test_accumulate_histogram(self):
        data = [[[[random.randint(0, 10) for i in range(2)]
                  for j in range(3)]
                 for k in range(4)]
                for r in range(5)]
        histograms = copy.deepcopy(data)
        for i in range(len(data)):
            for j in range(len(data[i])):
                histograms[i][j] = self.feature_histogram.accumulate_histogram(histograms[i][j])
                for k in range(1, len(data[i][j])):
                    for r in range(len(data[i][j][k])):
                        data[i][j][k][r] += data[i][j][k - 1][r]
                        self.assertTrue(data[i][j][k][r] == histograms[i][j][k][r])

    def test_calculate_histogram(self):
        histograms = self.feature_histogram.calculate_histogram(
            self.data_bin, self.grad_and_hess,
            self.bin_split_points, self.bin_sparse,
            node_map=self.node_map)

        his2 = [[[[0 for i in range(3)]
                  for j in range(6)]
                 for k in range(10)]
                for r in range(4)]
        for i in range(1000):
            grad, hess = self.grad_and_hess_list[i]
            id = self.node_map[self.data_insts[i][1][1]]
            for fid, bid in self.data_insts[i][0].features.get_all_data():
                his2[id][fid][bid][0] += grad
                his2[id][fid][bid][1] += hess
                his2[id][fid][bid][2] += 1

        for i in range(len(his2)):
            for j in range(len(his2[i])):
                his2[i][j] = self.feature_histogram.accumulate_histogram(his2[i][j])
                for k in range(len(his2[i][j])):
                    for r in range(len(his2[i][j][k])):
                        self.assertTrue(np.fabs(his2[i][j][k][r] - histograms[i][j][k][r]) < consts.FLOAT_ZERO)

    def test_aggregate_histogram(self):
        data1 = [[random.randint(0, 10) for i in range(2)] for j in range(3)]

        data2 = [[random.randint(0, 10) for i in range(2)] for j in range(3)]

        agg_histograms = self.feature_histogram.aggregate_histogram(data1, data2)
        for i in range(len(data1)):
            for j in range(len(data1[i])):
                data1[i][j] += data2[i][j]
                self.assertTrue(data1[i][j] == agg_histograms[i][j])


if __name__ == '__main__':
    unittest.main()
