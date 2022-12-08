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

from federatedml.feature.instance import Instance
from federatedml.optim.gradient.homo_lr_gradient import LogisticGradient, TaylorLogisticGradient
from federatedml.secureprotol import PaillierEncrypt


class TestHomoLRGradient(unittest.TestCase):
    def setUp(self):
        self.paillier_encrypt = PaillierEncrypt()
        self.paillier_encrypt.generate_key()
        self.gradient_operator = LogisticGradient()
        self.taylor_operator = TaylorLogisticGradient()

        self.X = np.array([[1, 2, 3, 4, 5], [3, 2, 4, 5, 1], [2, 2, 3, 1, 1, ]]) / 10
        self.X1 = np.c_[self.X, np.ones(3)]

        self.Y = np.array([[1], [1], [-1]])

        self.values = []
        for idx, x in enumerate(self.X):
            inst = Instance(inst_id=idx, features=x, label=self.Y[idx])
            self.values.append((idx, inst))

        self.values1 = []
        for idx, x in enumerate(self.X1):
            inst = Instance(inst_id=idx, features=x, label=self.Y[idx])
            self.values1.append((idx, inst))

        self.coef = np.array([2, 2.3, 3, 4, 2.1]) / 10
        self.coef1 = np.append(self.coef, [1])

    def test_gradient_length(self):
        fit_intercept = False
        grad = self.gradient_operator.compute_gradient(self.values, self.coef, 0, fit_intercept)
        self.assertEqual(grad.shape[0], self.X.shape[1])

        taylor_grad = self.taylor_operator.compute_gradient(self.values, self.coef, 0, fit_intercept)
        self.assertEqual(taylor_grad.shape[0], self.X.shape[1])
        self.assertTrue(np.sum(grad - taylor_grad) < 0.0001)

        fit_intercept = True
        grad = self.gradient_operator.compute_gradient(self.values, self.coef, 0, fit_intercept)
        self.assertEqual(grad.shape[0], self.X.shape[1] + 1)

        taylor_grad = self.taylor_operator.compute_gradient(self.values, self.coef, 0, fit_intercept)
        self.assertEqual(taylor_grad.shape[0], self.X.shape[1] + 1)

        self.assertTrue(np.sum(grad - taylor_grad) < 0.0001)


if __name__ == '__main__':
    unittest.main()
