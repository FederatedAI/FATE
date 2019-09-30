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

from arch.api.session import init
from federatedml.ftl.eggroll_computation.helper import distribute_compute_X_plus_Y
from federatedml.ftl.test.util import assert_matrix


class TestSum(unittest.TestCase):

    def test_distributed_calculate_X_plus_Y_1(self):
        X = np.array([[1., 2., 3.],
                      [14., 5., 6.],
                      [7., 8., 9.]])

        Y = np.array([[1], [-1], [1]])

        actual_X_plus_Y = X + Y
        X_plus_Y = distribute_compute_X_plus_Y(X, Y)
        assert_matrix(actual_X_plus_Y, X_plus_Y)

    def test_distributed_calculate_X_plus_Y_2(self):
        X = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])

        Z = np.array([[1., 2., 3.],
                      [1., 2., 3.],
                      [1., 2., 3.]])

        actual_X_plus_Z = X + Z
        X_plus_Z = distribute_compute_X_plus_Y(X, Z)
        assert_matrix(actual_X_plus_Z, X_plus_Z)


if __name__ == '__main__':
    init()
    unittest.main()
