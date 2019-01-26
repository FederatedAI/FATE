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
import unittest
import time

from federatedml.util.data_io import DenseFeatureReader
from federatedml.param.param import DataIOParam
from arch.api import eggroll
from federatedml.util import consts


class TestDataIO(unittest.TestCase):
    def setUp(self):
        eggroll.init("test_dataio" + str(int(time.time())))
        self.table = "dataio_table_test"
        self.namespace = "dataio_test"
        table = eggroll.parallelize([("a", "1,2,-1,0,0,5"), ("b", "4,5,6,0,1,2")], include_key=True)
        table.save_as(self.table, self.namespace)

        self.table2 = "dataio_table_test2"
        self.namespace2 = "dataio_test2"
        table2 = eggroll.parallelize([("a", '-1,,NA,NULL,null,2')], include_key=True)
        table2.save_as(self.table2, self.namespace2)

    def test_dense_output_format(self):
        dataio_param = DataIOParam()
        reader = DenseFeatureReader(dataio_param)
        data = reader.read_data(self.table, self.namespace).collect()
        result = dict(data)
        self.assertTrue(type(result['a']).__name__ == "Instance")
        self.assertTrue(type(result['b']).__name__ == "Instance")
        vala = result['a']
        features = vala.features
        weight = vala.weight
        label = vala.label
        self.assertTrue(np.abs(weight - 1.0) < consts.FLOAT_ZERO)
        self.assertTrue(type(features).__name__ == "ndarray")
        self.assertTrue(label == None)
        self.assertTrue(features.shape[0] == 6)
        self.assertTrue(features.dtype == "float64")

    def test_sparse_output_format(self):
        dataio_param = DataIOParam()
        dataio_param.output_format = "sparse"
        reader = DenseFeatureReader(dataio_param)
        data = reader.read_data(self.table, self.namespace).collect()
        result = dict(data)
        vala = result['a']
        features = vala.features
        self.assertTrue(type(features).__name__ == "SparseVector")
        self.assertTrue(len(features.sparse_vec) == 4)
        self.assertTrue(features.shape == 6)

    def test_missing_value_fill(self):
        dataio_param = DataIOParam()
        dataio_param.missing_fill = True
        dataio_param.output_format = "sparse"
        dataio_param.default_value = 100
        dataio_param.data_type = 'int'
        reader = DenseFeatureReader(dataio_param)
        data = reader.read_data(self.table2, self.namespace2).collect()
        result = dict(data)
        features = result['a'].features
        for i in range(1, 5):
            self.assertTrue(features.get_data(i) == 100)

    def test_with_label(self):
        dataio_param = DataIOParam()
        dataio_param.with_label = True
        dataio_param.label_idx = 2
        reader = DenseFeatureReader(dataio_param)
        data = reader.read_data(self.table, self.namespace).collect()
        result = dict(data)
        vala = result['a']
        label = vala.label
        features = vala.features
        self.assertTrue(label == -1)
        self.assertTrue(features.shape[0] == 5)


if __name__ == '__main__':
    unittest.main()
