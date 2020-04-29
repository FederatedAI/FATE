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
import uuid

from arch.api import session
import numpy as np


class TestStatistics(unittest.TestCase):
    def setUp(self):
        session.init((str(uuid.uuid1())))

    def test_standardized(self):
        from federatedml.statistic.correlation import hetero_pearson
        raw_data = np.random.rand(200, 100)
        expect = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)
        data_table = session.parallelize([row for row in raw_data], partition=10)
        n, standardized = hetero_pearson.HeteroPearson._standardized(data_table)
        standardized_data = np.array([row[1] for row in standardized.collect()])
        self.assertEqual(n, standardized_data.shape[0])
        self.assertEqual(raw_data.shape, standardized_data.shape)
        self.assertAlmostEqual(np.linalg.norm(standardized_data - expect), 0.0)


if __name__ == '__main__':
    unittest.main()
