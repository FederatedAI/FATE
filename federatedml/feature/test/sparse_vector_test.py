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

from federatedml.feature.sparse_vector import SparseVector


class TestSparseVector(unittest.TestCase):
    def setUp(self):
        pass

    def test_instance(self):
        indices = []
        data = []
        for i in range(1, 10):
            indices.append(i * i)
            data.append(i ** 3)

        shape = 100

        sparse_data = SparseVector(indices, data, shape)
        self.assertTrue(sparse_data.shape == shape and len(sparse_data.sparse_vec) == 9)
        self.assertTrue(sparse_data.count_zeros() == 91)
        self.assertTrue(sparse_data.count_non_zeros() == 9)

        for idx, val in zip(indices, data):
            self.assertTrue(sparse_data.get_data(idx) == val)
        for i in range(100):
            if i in indices:
                continue
            self.assertTrue(sparse_data.get_data(i, i ** 4) == i ** 4)

        self.assertTrue(dict(sparse_data.get_all_data()) == dict(zip(indices, data)))


if __name__ == '__main__':
    unittest.main()
