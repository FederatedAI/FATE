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

import math
import time
import unittest

import numpy as np

from arch.api import eggroll

eggroll.init("123")

from federatedml.feature.binning import QuantileBinning
from federatedml.feature.instance import Instance
from federatedml.param.param import FeatureBinningParam


class TestQuantileBinning(unittest.TestCase):
    def setUp(self):
        # eggroll.init("123")
        self.data_num = 1000
        self.feature_num = 200
        final_result = []
        numpy_array = []
        for i in range(self.data_num):
            tmp = np.random.randn(self.feature_num)
            inst = Instance(inst_id=i, features=tmp, label=0)
            tmp_pair = (str(i), inst)
            final_result.append(tmp_pair)
            numpy_array.append(tmp)
        table = eggroll.parallelize(final_result,
                                    include_key=True,
                                    partition=10)
        self.table = table
        self.numpy_table = np.array(numpy_array)
        self.cols = [1]

    def test_quantile_binning(self):

        compress_thres = 10000
        head_size = 5000
        error = 0.01
        bin_num = 10
        bin_param = FeatureBinningParam(method='quantile', compress_thres=compress_thres, head_size=head_size,
                                        error=error,
                                        bin_num=bin_num)
        quan_bin = QuantileBinning(bin_param)
        split_points = quan_bin.binning(self.table, cols=self.cols)
        for col_idx, col in enumerate(self.cols):
            bin_percent = [i * (1.0 / bin_num) for i in range(1, bin_num)]
            x = self.numpy_table[:, col]
            x = sorted(x)
            for bin_idx, percent in enumerate(bin_percent):
                min_rank = int(math.floor(percent * self.data_num - self.data_num * error))
                max_rank = int(math.ceil(percent * self.data_num + self.data_num * error))
                if min_rank < 0:
                    min_rank = 0
                if max_rank > len(x) - 1:
                    max_rank = len(x) - 1
                try:
                    self.assertTrue(x[min_rank] <= split_points[col_idx][bin_idx] <= x[max_rank])
                except:
                    print(x[min_rank], x[max_rank], split_points[col_idx][bin_idx])
                    found_index = x.index(split_points[col_idx][bin_idx])
                    print("min_rank: {}, found_rank: {}, max_rank: {}".format(
                        min_rank, found_index, max_rank
                    ))
                self.assertTrue(x[min_rank] <= split_points[col_idx][bin_idx] <= x[max_rank])

    def tearDown(self):
        self.table.destroy()


class TestQuantileBinningSpeed(unittest.TestCase):
    def setUp(self):
        # eggroll.init("123")
        self.data_num = 100000
        self.feature_num = 200
        final_result = []
        numpy_array = []
        for i in range(self.data_num):
            tmp = np.random.randn(self.feature_num)
            inst = Instance(inst_id=i, features=tmp, label=0)
            tmp_pair = (str(i), inst)
            final_result.append(tmp_pair)
            numpy_array.append(tmp)
        table = eggroll.parallelize(final_result,
                                    include_key=True,
                                    partition=10)
        self.table = table
        self.numpy_table = np.array(numpy_array)
        self.cols = [1, 2, 3]

    def test_quantile_binning(self):
        error = 0.01
        compress_thres = int(self.data_num / (self.data_num * error))

        head_size = 5000
        bin_num = 10
        bin_percent = [int(i * (100.0 / bin_num)) for i in range(1, bin_num)]

        bin_param = FeatureBinningParam(method='quantile', compress_thres=compress_thres, head_size=head_size, error=error,
                                 bin_num=bin_num)
        quan_bin = QuantileBinning(bin_param)
        t0 = time.time()
        split_points = quan_bin.binning(self.table, cols=self.cols)
        t1 = time.time()
        print('Spend time: {}'.format(t1 - t0))

        # collect and test numpy quantile speed
        local_table = self.table.collect()
        total_data = []
        for _, data_inst in local_table:
            total_data.append(data_inst.features)
        total_data = np.array(total_data)
        for col in self.cols:
            x = total_data[:, col]
            sk = np.percentile(x, bin_percent, interpolation="midpoint")
        t2 = time.time()
        print('collect and use numpy time: {}'.format(t2 - t1))

    def tearDown(self):
        self.table.destroy()


if __name__ == '__main__':
    unittest.main()
