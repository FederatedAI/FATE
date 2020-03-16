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

session.init("123")

from federatedml.feature.binning.bucket_binning import BucketBinning
from federatedml.feature.instance import Instance
from federatedml.param.feature_binning_param import FeatureBinningParam


class TestBucketBinning(unittest.TestCase):
    def setUp(self):
        # eggroll.init("123")
        self.data_num = 1000
        self.feature_num = 200
        self.bin_num = 10
        final_result = []
        numpy_array = []
        for i in range(self.data_num):
            if 100 < i < 500:
                continue
            tmp = i * np.ones(self.feature_num)
            inst = Instance(inst_id=i, features=tmp, label=i%2)
            tmp_pair = (str(i), inst)
            final_result.append(tmp_pair)
            numpy_array.append(tmp)
        table = session.parallelize(final_result,
                                    include_key=True,
                                    partition=10)

        header = ['x' + str(i) for i in range(self.feature_num)]

        self.table = table
        self.table.schema = {'header': header}

        self.numpy_table = np.array(numpy_array)
        self.cols = [1, 2]

    def test_bucket_binning(self):
        bin_param = FeatureBinningParam(bin_num=self.bin_num, bin_indexes=self.cols)
        bucket_bin = BucketBinning(bin_param)
        split_points = bucket_bin.fit_split_points(self.table)
        split_point = list(split_points.values())[0]
        for kth, s_p in enumerate(split_point):
            expect_s_p = (self.data_num - 1) / self.bin_num * (kth + 1)
            self.assertEqual(s_p, expect_s_p)
        bucket_bin.cal_local_iv(self.table)
        for col_name, iv_attr in bucket_bin.bin_results.all_cols_results.items():
            # print('col_name: {}, iv: {}, woe_array: {}'.format(col_name, iv_attr.iv, iv_attr.woe_array))
            assert abs(iv_attr.iv - 0.00364386529386804) < 1e-6

    def tearDown(self):
        self.table.destroy()


if __name__ == '__main__':
    unittest.main()
