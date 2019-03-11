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
# from arch.api import eggroll
from federatedml.optim import L1Updater
from federatedml.optim import L2Updater
import numpy as np
import unittest


class TestUpdater(unittest.TestCase):
    def setUp(self):
        alpha = 0.5
        learning_rate = 0.1
        self.l1_updater = L1Updater(alpha, learning_rate)
        self.l2_updater = L2Updater(alpha, learning_rate)

        self.coef_ = np.array([1, -2, 3, -4, 5, -6, 7, -8, 9])
        self.gradient = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

        # l2 regular
        self.l2_loss_norm = 0.5 * alpha * np.sum(np.array([i * i for i in self.coef_]))
        self.l2_update_coef = self.coef_ - self.gradient - learning_rate * alpha * self.coef_

        # l1 regular
        self.l1_loss_norm = 22.5
        self.l1_update_coef = [0, -2.95, 1.95, -4.95, 3.95, -6.95, 5.95, -8.95, 7.95]

    def test_l2updater(self):
        loss_norm = self.l2_updater.loss_norm(self.coef_)
        self.assertEqual(loss_norm, self.l2_loss_norm)

        l2_update_coef = self.l2_updater.update_coef(self.coef_, self.gradient)
        self.assertListEqual(list(l2_update_coef), list(self.l2_update_coef))

    def test_l1updater(self):
        loss_norm = self.l1_updater.loss_norm(self.coef_)
        self.assertEqual(loss_norm, self.l1_loss_norm)

        l1_update_coef = self.l1_updater.update_coef(self.coef_, self.gradient)
        self.assertListEqual(list(l1_update_coef), list(self.l1_update_coef))


if __name__ == "__main__":
    unittest.main()
