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

import numpy as np
import random
import unittest
from arch.api import eggroll
from federatedml.util import consts
from federatedml.util import Statistics


class TestStatistics(unittest.TestCase):
    def setUp(self):
        eggroll.init("test_statistics")
        self.data = []
        for i in range(100):
            self.data.append(random.random())
        self.table_1p = eggroll.parallelize(self.data, partition=1)
        self.table_4p = eggroll.parallelize(self.data, partition=4)

    def test_median_list(self):
        mid = np.median(self.data)
        statis = Statistics()
        sta_mid = statis.median(self.data)
        self.assertTrue(np.abs(mid - sta_mid) < consts.FLOAT_ZERO)

    def test_median_dtable(self):
        mid = np.median(self.data)
        statis = Statistics()
        sta_mid = statis.median(self.table_1p)
        self.assertTrue(np.abs(mid - sta_mid) < consts.FLOAT_ZERO)

    def test_mean_list(self):
        mean = np.mean(self.data)
        statis = Statistics()
        sta_mean = statis.mean(self.data)
        self.assertTrue(np.abs(mean - sta_mean) < consts.FLOAT_ZERO)

    def test_mean_dtable(self):
        mean = np.mean(self.data)
        statis = Statistics()
        sta_mean = statis.mean(self.table_4p)
        self.assertTrue(np.abs(mean - sta_mean) < consts.FLOAT_ZERO)


if __name__ == '__main__':
    unittest.main()
