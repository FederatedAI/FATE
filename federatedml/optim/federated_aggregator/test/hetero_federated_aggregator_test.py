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

import random
import unittest
import numpy as np

from arch.api import eggroll
from federatedml.optim.federated_aggregator import HeteroFederatedAggregator


class TestHeteroFederatedAggregator(unittest.TestCase):
    def setUp(self):
        # for test_aggregate_add
        eggroll.init("test_hetero_federated_aggregator")
        self.size = 10
        self.table_a = eggroll.parallelize(range(self.size))
        self.table_b = eggroll.parallelize(list(range(self.size)))
        self.add_a_b = [i * 2 for i in range(self.size)]

        # for test_aggregate_mean
        self.table_d_tuple = eggroll.parallelize([(i, i + 1) for i in range(self.size)])
        self.reduce_a = np.sum(list(range(self.size))) / self.size * 1.0
        self.reduce_d_tuple = [np.sum(list(range(self.size))) / self.size * 1.0,
                               np.sum(list(range(self.size + 1))) / self.size * 1.0]

        # for test_separate
        self.separate_data = list(range(self.size))
        self.separate_size_list = [int(0.1 * self.size), int(0.2 * self.size), int(0.3 * self.size),
                                   int(0.4 * self.size)]
        self.separate_result = []
        cur_index = 0
        for i in range(len(self.separate_size_list)):
            self.separate_result.append(self.separate_data[cur_index:cur_index + self.separate_size_list[i]])
            cur_index += self.separate_size_list[i]

        # for test_aggregate_add_square
        this_size = 10000
        list_a = [random.randint(0, 1000) for _ in range(this_size)]
        list_b = [random.randint(0, 1000) for _ in range(this_size)]
        self.table_list_a = eggroll.parallelize(list_a)
        self.table_list_b = eggroll.parallelize(list_b)
        self.table_list_a_square = eggroll.parallelize([np.square(i) for i in list_a])
        self.table_list_b_square = eggroll.parallelize([np.square(i) for i in list_b])

        self.list_add_square_result = list(np.sort(np.array([np.square(i + j) for (i, j) in zip(list_a, list_b)])))

    def test_aggregate_add(self):
        table_add_res = HeteroFederatedAggregator.aggregate_add(self.table_a, self.table_b)

        res = []
        for iterater in table_add_res.collect():
            res.append(iterater[1])

        res = np.sort(np.array(res))
        self.assertListEqual(self.add_a_b, list(res))

    def test_aggreagte_mean(self):
        res = HeteroFederatedAggregator.aggregate_mean(self.table_a)
        self.assertEqual(res, self.reduce_a)
        res = HeteroFederatedAggregator.aggregate_mean(self.table_d_tuple)
        self.assertListEqual(list(res), self.reduce_d_tuple)

    def test_separate(self):
        res = HeteroFederatedAggregator.separate(self.separate_data, self.separate_size_list)
        self.assertListEqual(res, self.separate_result)

    def test_aggregate_add_square(self):
        res = HeteroFederatedAggregator.aggregate_add_square(self.table_list_a, self.table_list_b,
                                                             self.table_list_a_square,
                                                             self.table_list_b_square).collect()
        res_to_list = []
        for iterator in res:
            res_to_list.append(iterator[1])

        res = list(np.sort(np.array(res_to_list)))
        self.assertListEqual(self.list_add_square_result, res)

    # def tearDown(self):


if __name__ == '__main__':
    unittest.main()
