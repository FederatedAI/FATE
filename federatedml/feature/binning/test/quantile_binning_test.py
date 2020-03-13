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
#
import unittest
import uuid

import numpy as np

from arch.api import session
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.util import consts

bin_num = 10

TEST_LARGE_DATA = False

# job_id = str(uuid.uuid1())
# session.init(job_id, 1)


class TestQuantileBinning(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        session.init(self.job_id)

    def test_binning_correctness(self):
        bin_obj = self._bin_obj_generator()
        small_table = self.gen_data(10000, 50, 2)
        split_points = bin_obj.fit_split_points(small_table)
        expect_split_points = list((range(1, bin_num)))
        expect_split_points = [float(x) for x in expect_split_points]

        for _, s_ps in split_points.items():
            s_ps = s_ps.tolist()
            self.assertListEqual(s_ps, expect_split_points)

    def test_large_binning(self):
        if TEST_LARGE_DATA:
            bin_obj = self._bin_obj_generator()
            small_table = self.gen_data(100000, 1000, 48, use_random=True)
            _ = bin_obj.fit_split_points(small_table)

    def test_sparse_data(self):
        feature_num = 50
        bin_obj = self._bin_obj_generator()
        small_table = self.gen_data(10000, feature_num, 2, is_sparse=True)
        split_points = bin_obj.fit_split_points(small_table)
        expect_split_points = list((range(1, bin_num)))
        expect_split_points = [float(x) for x in expect_split_points]

        for feature_name, s_ps in split_points.items():
            if int(feature_name) >= feature_num:
                continue
            s_ps = s_ps.tolist()
            self.assertListEqual(s_ps, expect_split_points)

    def test_abnormal(self):
        abnormal_list = [3, 4]
        bin_obj = self._bin_obj_generator(abnormal_list=abnormal_list, this_bin_num=bin_num - len(abnormal_list))
        small_table = self.gen_data(10000, 50, 2)
        split_points = bin_obj.fit_split_points(small_table)
        expect_split_points = list((range(1, bin_num)))
        expect_split_points = [float(x) for x in expect_split_points if x not in abnormal_list]

        for _, s_ps in split_points.items():
            s_ps = s_ps.tolist()
            self.assertListEqual(s_ps, expect_split_points)

    def _bin_obj_generator(self, abnormal_list: list = None, this_bin_num=bin_num):

        bin_param = FeatureBinningParam(method='quantile', compress_thres=consts.DEFAULT_COMPRESS_THRESHOLD,
                                        head_size=consts.DEFAULT_HEAD_SIZE,
                                        error=consts.DEFAULT_RELATIVE_ERROR,
                                        bin_indexes=-1,
                                        bin_num=this_bin_num)
        bin_obj = QuantileBinning(bin_param, abnormal_list=abnormal_list)
        return bin_obj

    def gen_data(self, data_num, feature_num, partition, is_sparse=False, use_random=False):
        data = []
        shift_iter = 0
        header = [str(i) for i in range(feature_num)]

        for data_key in range(data_num):
            value = data_key % bin_num
            if value == 0:
                if shift_iter % bin_num == 0:
                    value = bin_num - 1
                shift_iter += 1
            if not is_sparse:
                if not use_random:
                    features = value * np.ones(feature_num)
                else:
                    features = np.random.random(feature_num)
                inst = Instance(inst_id=data_key, features=features, label=data_key % 2)

            else:
                if not use_random:
                    features = value * np.ones(feature_num)
                else:
                    features = np.random.random(feature_num)
                data_index = [x for x in range(feature_num)]
                sparse_inst = SparseVector(data_index, data=features, shape=10 * feature_num)
                inst = Instance(inst_id=data_key, features=sparse_inst, label=data_key % 2)
                header = [str(i) for i in range(feature_num * 10)]

            data.append((data_key, inst))
        result = session.parallelize(data, include_key=True, partition=partition)
        result.schema = {'header': header}
        return result

    # def tearDown(self):
    #     session.stop()
        # try:
        #     session.cleanup("*", self.job_id, True)
        # except EnvironmentError:
        #     pass
        # try:
        #     session.cleanup("*", self.job_id, False)
        # except EnvironmentError:
        #     pass


if __name__ == '__main__':
    unittest.main()
