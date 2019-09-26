#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import random
import unittest

import numpy as np

from arch.api import session
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector


class TestHeteroSecureBoostGuest(unittest.TestCase):
    def setUp(self):
        self.data = []
        for i in range(100):
            dict = {}
            indices = []
            data = []
            for j in range(40):
                idx = random.randint(0, 49)
                if idx in dict:
                    continue
                dict[idx] = 1
                val = random.random()
                indices.append(idx)
                data.append(val)

            sparse_vec = SparseVector(indices, data, 50)
            self.data.append((str(i), Instance(features=sparse_vec, label=i % 2)))

        self.table = session.parallelize(self.data, include_key=True)
        self.table.schema = {"header": ["fid" + str(i) for i in range(50)]}

    def test_join(self):
        table1 = self.table
        table2 = self.table.mapValues(lambda x: None)
        local_table2 = table2.collect()
        table_sid = []
        for k, _ in local_table2:
            table_sid.append(k)

        data_sids = np.array(table_sid)
        train_sids_table = [(str(x), 1) for x in data_sids]
        table2 = session.parallelize(train_sids_table,
                                     include_key=True,
                                     partition=table1._partitions)

        table3 = table1.join(table2, lambda x, y: x)
        self.assertTrue(table3.count() == table1.count())


if __name__ == '__main__':
    session.init("jobid")
    unittest.main()