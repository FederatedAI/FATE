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


import time
import unittest
import uuid

import numpy as np
from fate_arch.session import computing_session as session

from federatedml.feature.instance import Instance
from federatedml.util.sample_weight import SampleWeight


class TestSampleWeight(unittest.TestCase):
    def setUp(self):
        session.init("test_sample_weight_" + str(uuid.uuid1()))
        self.class_weight = {"0": 2, "1": 3}
        data = []
        for i in range(1, 11):
            label = 1 if i % 5 == 0 else 0
            instance = Instance(inst_id=i, features=np.random.random(3), label=label)
            data.append((i, instance))
        schema = {"header": ["x0", "x1", "x2"],
                  "sid": "id", "label_name": "y"}
        self.table = session.parallelize(data, include_key=True, partition=8)
        self.table.schema = schema
        self.sample_weight_obj = SampleWeight()

    def test_get_class_weight(self):
        class_weight = self.sample_weight_obj.get_class_weight(self.table)
        c_class_weight = {"1": 10 / 4, "0": 10 / 16}
        self.assertDictEqual(class_weight, c_class_weight)

    def test_replace_weight(self):
        instance = self.table.first()
        weighted_instance = self.sample_weight_obj.replace_weight(instance[1], self.class_weight)
        self.assertEqual(weighted_instance.weight, self.class_weight[str(weighted_instance.label)])

    def test_assign_sample_weight(self):
        weighted_table = self.sample_weight_obj.assign_sample_weight(self.table, self.class_weight, None, False)
        weighted_table.mapValues(lambda v: self.assertEqual(v.weight, self.class_weight[str(v.label)]))

    def test_get_weight_loc(self):
        c_loc = 2
        loc = self.sample_weight_obj.get_weight_loc(self.table, "x2")
        self.assertEqual(loc, c_loc)

    def tearDown(self):
        session.stop()


if __name__ == '__main__':
    unittest.main()
