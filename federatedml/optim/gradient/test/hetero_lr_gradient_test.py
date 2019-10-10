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
from federatedml.optim.gradient import hetero_lr_gradient_and_loss
from federatedml.secureprotol import PaillierEncrypt
from federatedml.linear_model.linear_model_weight import LinearModelWeights


class TestHeteroLogisticGradient(unittest.TestCase):
    def setUp(self):
        self.paillier_encrypt = PaillierEncrypt()
        self.paillier_encrypt.generate_key()
        # self.hetero_lr_gradient = HeteroLogisticGradient(self.paillier_encrypt)
        self.hetero_lr_gradient = hetero_lr_gradient_and_loss.Guest()

        size = 10
        self.wx = session.parallelize([self.paillier_encrypt.encrypt(i) for i in range(size)])
        self.en_sum_wx_square = session.parallelize([self.paillier_encrypt.encrypt(np.square(i)) for i in range(size)])
        self.w = [i for i in range(size)]
        self.data_inst = session.parallelize(
            [Instance(features=[1 for _ in range(size)], label=pow(-1, i % 2)) for i in range(size)], partition=1)

        # test fore_gradient
        self.fore_gradient_local = [-0.5, 0.75, 0, 1.25, 0.5, 1.75, 1, 2.25, 1.5, 2.75]
        # test gradient
        self.gradient = [1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125]
        self.gradient_fit_intercept = [1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125]

        self.loss = 4.505647

    # def test_compute_fore_gradient(self):
    #     # fore_gradient = self.hetero_lr_gradient.compute_and_aggregate_forwards(self.data_inst, self.wx)
    #     model_weights = LinearModelWeights(l=self.w, fit_intercept=False)
    #
    #     class EncryptedCalculator(object):
    #         encrypter = self.paillier_encrypt
    #
    #         def encrypt_row(self, row):
    #             return np.array([self.encrypter.encrypt(row)])
    #
    #         def encrypt(self, input_data):
    #             return input_data.mapValues(self.encrypt_row)
    #
    #     encrypted_calculator = [EncryptedCalculator()]
    #     batch_index = 0
    #     fore_gradient = self.hetero_lr_gradient.compute_and_aggregate_forwards(self.data_inst,
    #                                                                            model_weights,
    #                                                                            encrypted_calculator,
    #                                                                            batch_index)
    #
    #     fore_gradient_local = [self.paillier_encrypt.decrypt(iterator[1]) for iterator in fore_gradient.collect()]
    #
    #     self.assertListEqual(fore_gradient_local, self.fore_gradient_local)

    # def test_compute_gradient(self):
    #     fore_gradient = self.hetero_lr_gradient.compute_fore_gradient(self.data_inst, self.wx)
    #
    #     gradient = self.hetero_lr_gradient.compute_gradient(self.data_inst, fore_gradient, fit_intercept=False)
    #     de_gradient = [self.paillier_encrypt.decrypt(iterator) for iterator in gradient]
    #     self.assertListEqual(de_gradient, self.gradient)
    #
    #     gradient = self.hetero_lr_gradient.compute_gradient(self.data_inst, fore_gradient, fit_intercept=True)
    #     de_gradient = [self.paillier_encrypt.decrypt(iterator) for iterator in gradient]
    #     self.assertListEqual(de_gradient, self.gradient_fit_intercept)
    #
    # def test_compute_gradient_and_loss(self):
    #     fore_gradient = self.hetero_lr_gradient.compute_fore_gradient(self.data_inst, self.wx)
    #     gradient, loss = self.hetero_lr_gradient.compute_gradient_and_loss(self.data_inst, fore_gradient, self.wx,
    #                                                                        self.en_sum_wx_square, False)
    #     de_gradient = [self.paillier_encrypt.decrypt(i) for i in gradient]
    #     self.assertListEqual(de_gradient, self.gradient)
    #
    #     diff_loss = np.abs(self.loss - self.paillier_encrypt.decrypt(loss))
    #     self.assertLess(diff_loss, 1e-5)


if __name__ == "__main__":
    session.init("1111")
    unittest.main()
