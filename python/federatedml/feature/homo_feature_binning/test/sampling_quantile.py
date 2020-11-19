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


import bisect
import unittest

import numpy as np


# from federatedml.feature.quantile import Quantile


class TestInstance(unittest.TestCase):
    def setUp(self):
        self.e = 0.0001
        self.node1 = np.random.randint(0, 1000, 10000)
        self.node2 = np.random.randint(900, 2000, 20000)
        self.node3 = np.random.randint(1800, 3000, 30000)
        all_data = np.append(self.node1, self.node2)
        all_data = np.append(all_data, self.node3)
        self.all_data = np.array(sorted(all_data))
        self.n = len(self.all_data)
        self.k = 3
        self.is_sampled = False

    def _sample_data(self, node_data):
        sorted_data = np.array(sorted(node_data))
        # print(f"sorted_data: {sorted_data}")
        data_len = len(sorted_data)
        idx_list = np.array(list(range(data_len)))
        if data_len > (self.n / np.sqrt(self.k)):
            p = 1 / (self.e * data_len)
        else:
            p = np.sqrt(self.k) / (self.e * self.n)
        p = min(1.0, p)

        # print(f"p: {p}, choice_data: {int(p * data_len)}")
        sampled_idx = np.random.choice(data_len, int(p * data_len), replace=False)
        sampled_idx = sorted(sampled_idx)
        sampled_data = sorted_data[sampled_idx]
        # print(sampled_data, sampled_idx)
        return sampled_data, sampled_idx, p

    def _estimate_rank(self, value, sampled_data, sampled_rank, p):
        idx = bisect.bisect_left(sampled_data, value)
        if idx == 0:
            esit_rank = 0
        else:
            esit_rank = sampled_rank[idx - 1] + (1 / p)
        return esit_rank

    def test_estimate_rank(self):
        if not self.is_sampled:
            self.sampled_node1, self.sampled_idx1, self.p1 = self._sample_data(self.node1)
            self.sampled_node2, self.sampled_idx2, self.p2 = self._sample_data(self.node2)
            self.sampled_node3, self.sampled_idx3, self.p3 = self._sample_data(self.node3)
            self.is_sampled = True
        value = 999
        true_rank = bisect.bisect_left(self.all_data, value)

        esti_rank = self._estimate_rank(value, self.sampled_node1, self.sampled_idx1, self.p1) + \
                    self._estimate_rank(value, self.sampled_node2, self.sampled_idx2, self.p2) + \
                    self._estimate_rank(value, self.sampled_node3, self.sampled_idx3, self.p3)
        esti_rank = int(esti_rank)
        # error_range = self.e *
        print(f"true_rank: {true_rank}, esti_rank: {esti_rank}")

    def test_estimate_quantile(self):
        quantile = 0.7
        value = np.quantile(self.all_data, quantile)

        rank = quantile * self.n
        print(f"true_value: {value}")


if __name__ == '__main__':
    unittest.main()
