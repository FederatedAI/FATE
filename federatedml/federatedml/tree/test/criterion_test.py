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
from federatedml.tree import XgboostCriterion
from federatedml.util import consts


class TestXgboostCriterion(unittest.TestCase):
    def setUp(self):
        self.reg_lambda = 0.3
        self.criterion = XgboostCriterion(reg_lambda=self.reg_lambda)

    def test_init(self):
        self.assertTrue(np.fabs(self.criterion.reg_lambda - self.reg_lambda) < consts.FLOAT_ZERO)

    def test_split_gain(self):
        node = [0.5, 0.6]
        left = [0.1, 0.2]
        right = [0.4, 0.4]
        gain_all = node[0] * node[0] / (node[1] + self.reg_lambda)
        gain_left = left[0] * left[0] / (left[1] + self.reg_lambda)
        gain_right = right[0] * right[0] / (right[1] + self.reg_lambda)
        split_gain = gain_left + gain_right - gain_all
        self.assertTrue(np.fabs(self.criterion.split_gain(node, left, right) - split_gain) < consts.FLOAT_ZERO)

    def test_node_gain(self):
        grad = 0.5
        hess = 6
        gain = grad * grad / (hess + self.reg_lambda)
        self.assertTrue(np.fabs(self.criterion.node_gain(grad, hess) - gain) < consts.FLOAT_ZERO)

    def test_node_weight(self):
        grad = 0.5
        hess = 6
        weight = -grad / (hess + self.reg_lambda)
        self.assertTrue(np.fabs(self.criterion.node_weight(grad, hess) - weight) < consts.FLOAT_ZERO)


if __name__ == '__main__':
    unittest.main()
