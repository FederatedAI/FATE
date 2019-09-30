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
from federatedml.ftl.eggroll_computation.helper import distribute_compute_avg_XY, distribute_compute_sum_XY, distribute_compute_XY, distribute_compute_XY_plus_Z
from federatedml.ftl.test.util import assert_matrix, assert_array


class TestSum(unittest.TestCase):

    def test_distributed_calculate_XY_1(self):
        print("--- test_distributed_calculate_XY_1 ---")
        # X has shape (4, 3)
        X = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.],
                      [10, 11, 12]])

        # Y has shape (4, 1)
        Y = np.array([[2], [1], [-1], [1]])

        actual_XY = X * Y
        print(actual_XY, actual_XY.shape)
        XY = distribute_compute_XY(X, Y)
        assert_matrix(actual_XY, XY)

    def test_distributed_calculate_XY_2(self):
        print("--- test_distributed_calculate_XY_2 ---")
        # X has shape (4, 3, 3)
        X = np.random.rand(4, 3, 3)

        # Y has shape (4, 1, 1)
        Y = np.random.rand(4, 1, 1)

        actual_XY = X * Y
        print(actual_XY, actual_XY.shape)
        XY = distribute_compute_XY(X, Y)
        assert_matrix(actual_XY, XY)

    def test_distributed_calculate_avg_XY_1(self):
        print("--- test_distributed_calculate_avg_XY_1 ---")

        X = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])

        Y = np.array([[1], [-1], [1]])

        actual_avg_XY = np.mean(X * Y, axis=0)
        avg_XY = distribute_compute_avg_XY(X, Y)
        assert_array(actual_avg_XY, avg_XY)

    def test_distributed_calculate_avg_XY_2(self):
        print("--- test_distributed_calculate_avg_XY_2 ---")

        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=np.float64)
        Y = np.array([[1], [-1], [1]])
        Y = np.tile(Y, (1, X.shape[-1]))

        actual1 = np.sum(Y * X, axis=0) / len(Y)
        actual2 = np.sum(X * Y, axis=0) / len(Y)
        predict1 = distribute_compute_avg_XY(X, Y)
        predict2 = distribute_compute_avg_XY(Y, X)
        assert_array(actual1, predict1)
        assert_array(actual2, predict2)

    def test_distributed_calculate_sum_XY(self):
        print("--- test_distributed_calculate_sum_XY ---")

        X = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])

        Y = np.array([[1], [-1], [1]])

        actual_sum_XY = np.sum(X * Y, axis=0)
        sum_XY = distribute_compute_sum_XY(X, Y)
        assert_array(actual_sum_XY, sum_XY)

    def test_distributed_compute_XY_plus_Z(self):
        print("--- test_distributed_compute_XY_plus_Z ---")

        X = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])

        Y = np.array([[1], [-1], [1]])

        Z = np.array([[1., 2., 3.],
                      [1., 2., 3.],
                      [1., 2., 3.]])

        actual_XY_plus_Z = X * Y + Z
        XY_plus_Z = distribute_compute_XY_plus_Z(X, Y, Z)
        assert_matrix(actual_XY_plus_Z, XY_plus_Z)


if __name__ == '__main__':
    init()
    unittest.main()
