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

eggroll.init("123")

from federatedml.feature.binning.bucket_binning import BucketBinning
from federatedml.feature.instance import Instance
from federatedml.param.param import FeatureBinningParam


class TestBucketBinning(unittest.TestCase):
    def setUp(self):
        # eggroll.init("123")
        self.data_num = 1000
        self.feature_num = 200
        self.bin_num = 10
        final_result = []
        numpy_array = []
        for i in range(self.data_num):
            tmp = i * np.ones(self.feature_num)
            inst = Instance(inst_id=i, features=tmp, label=0)
            tmp_pair = (str(i), inst)
            final_result.append(tmp_pair)
            numpy_array.append(tmp)
        table = eggroll.parallelize(final_result,
                                    include_key=True,
                                    partition=10)

        header = ['x' + str(i) for i in range(self.feature_num)]

        self.table = table
        self.table.schema = {'header': header}

        self.numpy_table = np.array(numpy_array)
        self.cols = ['x1', 'x2']

    def test_bucket_binning(self):
        bin_param = FeatureBinningParam(bin_num=self.bin_num, cols=self.cols)
        bucket_bin = BucketBinning(bin_param)
        split_points = bucket_bin.fit_split_points(self.table)
        print(split_points)

    def tearDown(self):
        self.table.destroy()


if __name__ == '__main__':
    unittest.main()
