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
import random
from arch.api import session
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.feature.instance import Instance
# from federatedml.feature.quantile import Quantile
from federatedml.feature.sparse_vector import SparseVector
from federatedml.param.feature_binning_param import FeatureBinningParam


class TestInstance(unittest.TestCase):
    # def setUp(self):
    #     eggroll.init("test_instance")
    #     dense_inst = []
    #     headers = ['x' + str(i) for i in range(20)]
    #     for i in range(100):
    #         inst = Instance(features=(i % 16 * np.ones(20)))
    #         dense_inst.append((i, inst))
    #     self.dense_table = eggroll.parallelize(dense_inst, include_key=True, partition=2)
    #     self.dense_table.schema = {'header': headers}
    #     sparse_inst = []
    #     col_zero = []
    #     for i in range(100):
    #         indices = []
    #         data = []
    #         for j in range(20):
    #             val = ((i + 5) ** 3 + (j + 1) ** 4) % 16
    #             if val > 0:
    #                 indices.append(j)
    #                 data.append(val)
    #             if j == 0:
    #                 col_zero.append(val)
    #         sparse_vec = SparseVector(indices, data, 20)
    #         inst = Instance(features=sparse_vec)
    #         sparse_inst.append((i, inst))
    #
    #     self.sparse_inst = sparse_inst
    #     self.sparse_table = eggroll.parallelize(sparse_inst, include_key=True, partition=1)
    #     self.sparse_table.schema = {'header': headers}

    def setUp(self):
        session.init("test_instance")

        dense_inst = []
        headers = ['x' + str(i) for i in range(20)]
        for i in range(100):
            inst = Instance(features=(i % 16 * np.ones(20)))
            dense_inst.append((i, inst))
        self.dense_table = session.parallelize(dense_inst, include_key=True, partition=2)
        self.dense_table.schema = {'header': headers}

        self.sparse_inst = []
        for i in range(100):
            dict = {}
            indices = []
            data = []
            for j in range(20):
                idx = random.randint(0, 29)
                if idx in dict:
                    continue
                dict[idx] = 1
                val = random.random()
                indices.append(idx)
                data.append(val)

            sparse_vec = SparseVector(indices, data, 30)
            self.sparse_inst.append((i, Instance(features=sparse_vec)))

        self.sparse_table = session.parallelize(self.sparse_inst, include_key=True)
        self.sparse_table.schema = {"header": ["fid" + str(i) for i in range(30)]}
        # self.sparse_table = eggroll.parallelize(sparse_inst, include_key=True, partition=1)

    """
    def test_dense_quantile(self):
        data_bin, bin_splitpoints, bin_sparse = Quantile.convert_feature_to_bin(self.dense_table, "bin_by_sample_data",
                                                                                bin_num=4)
        bin_result = dict([(key, inst.features) for key, inst in data_bin.collect()])
        for i in range(100):
            self.assertTrue((bin_result[i] == np.ones(20, dtype='int') * ((i % 16) // 4)).all())
            if i < 20:
                self.assertTrue((bin_splitpoints[i] == np.asarray([3, 7, 11, 15], dtype='int')).all())

        data_bin, bin_splitpoints, bin_sparse = Quantile.convert_feature_to_bin(self.dense_table, "bin_by_data_block",
                                                                                bin_num=4)
        for i in range(20):
            self.assertTrue(bin_splitpoints[i].shape[0] <= 4)

    def test_sparse_quantile(self):
        data_bin, bin_splitpoints, bin_sparse = Quantile.convert_feature_to_bin(self.sparse_table, "bin_by_sample_data",
                                                                                bin_num=4)
        bin_result = dict([(key, inst.features) for key, inst in data_bin.collect()])
        for i in range(20):
            self.assertTrue(len(self.sparse_inst[i][1].features.sparse_vec) == len(bin_result[i].sparse_vec))

    """

    def test_new_sparse_quantile(self):
        param_obj = FeatureBinningParam(bin_num=4)
        binning_obj = QuantileBinning(param_obj)
        binning_obj.fit_split_points(self.sparse_table)
        data_bin, bin_splitpoints, bin_sparse = binning_obj.convert_feature_to_bin(self.sparse_table)
        bin_result = dict([(key, inst.features) for key, inst in data_bin.collect()])
        for i in range(20):
            self.assertTrue(len(self.sparse_inst[i][1].features.sparse_vec) == len(bin_result[i].sparse_vec))

    def test_new_dense_quantile(self):
        param_obj = FeatureBinningParam(bin_num=4)
        binning_obj = QuantileBinning(param_obj)
        binning_obj.fit_split_points(self.dense_table)
        data_bin, bin_splitpoints, bin_sparse = binning_obj.convert_feature_to_bin(self.dense_table)
        bin_result = dict([(key, inst.features) for key, inst in data_bin.collect()])
        # print(bin_result)
        for i in range(100):
            self.assertTrue((bin_result[i] == np.ones(20, dtype='int') * ((i % 16) // 4)).all())
            if i < 20:
                # col_name = 'x' + str(i)
                col_idx = i
                split_point = np.array(bin_splitpoints[col_idx])
                self.assertTrue((split_point == np.asarray([3, 7, 11, 15], dtype='int')).all())

        for split_points in bin_splitpoints:
            self.assertTrue(len(split_points) <= 4)


if __name__ == '__main__':
    unittest.main()
