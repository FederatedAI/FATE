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
from federatedml.feature.sparse_vector import SparseVector
from federatedml.optim.gradient import hetero_linear_model_gradient
from federatedml.optim.gradient import hetero_lr_gradient_and_loss
from federatedml.secureprotol import PaillierEncrypt


class TestHeteroLogisticGradient(unittest.TestCase):
    def setUp(self):
        self.paillier_encrypt = PaillierEncrypt()
        self.paillier_encrypt.generate_key()
        # self.hetero_lr_gradient = HeteroLogisticGradient(self.paillier_encrypt)
        self.hetero_lr_gradient = hetero_lr_gradient_and_loss.Guest()

        size = 10
        self.en_wx = session.parallelize([self.paillier_encrypt.encrypt(i) for i in range(size)])
        # self.en_wx = session.parallelize([self.paillier_encrypt.encrypt(i) for i in range(size)])

        self.en_sum_wx_square = session.parallelize([self.paillier_encrypt.encrypt(np.square(i)) for i in range(size)])
        self.wx = np.array([i for i in range(size)])
        self.w = self.wx / np.array([1 for _ in range(size)])
        self.data_inst = session.parallelize(
            [Instance(features=np.array([1 for _ in range(size)]), label=pow(-1, i % 2)) for i in range(size)], partition=1)

        # test fore_gradient
        self.fore_gradient_local = [-0.5, 0.75, 0, 1.25, 0.5, 1.75, 1, 2.25, 1.5, 2.75]
        # test gradient
        self.gradient = [1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125]
        self.gradient_fit_intercept = [1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125]

        self.loss = 4.505647

    def test_compute_partition_gradient(self):
        fore_gradient = self.en_wx.join(self.data_inst, lambda wx, d: 0.25 * wx - 0.5 * d.label)
        sparse_data = self._make_sparse_data()
        for fit_intercept in [True, False]:
            dense_result = hetero_linear_model_gradient.compute_gradient(self.data_inst, fore_gradient, fit_intercept)
            dense_result = [self.paillier_encrypt.decrypt(iterator) for iterator in dense_result]
            if fit_intercept:
                self.assertListEqual(dense_result, self.gradient_fit_intercept)
            else:
                self.assertListEqual(dense_result, self.gradient)
            sparse_result = hetero_linear_model_gradient.compute_gradient(sparse_data, fore_gradient, fit_intercept)
            sparse_result = [self.paillier_encrypt.decrypt(iterator) for iterator in sparse_result]
            self.assertListEqual(dense_result, sparse_result)

    def _make_sparse_data(self):
        def trans_sparse(instance):
            dense_features = instance.features
            indices = [i for i in range(len(dense_features))]
            sparse_features = SparseVector(indices=indices, data=dense_features, shape=len(dense_features))
            return Instance(inst_id=None,
                            features=sparse_features,
                            label=instance.label)

        return self.data_inst.mapValues(trans_sparse)


if __name__ == "__main__":
    session.init("1111")
    unittest.main()
