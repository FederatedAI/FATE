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

import numpy as np
from fate_arch.session import computing_session as session

from federatedml.feature.instance import Instance
from federatedml.util.label_transform import LabelTransformer


class TestLabelTransform(unittest.TestCase):
    def setUp(self):
        session.init("test_label_transform_" + str(uuid.uuid1()))
        self.label_encoder = {"yes": 1, "no": 0}
        self.predict_label_encoder = {1: "yes", 0: "no"}
        data = []
        for i in range(1, 11):
            label = "yes" if i % 5 == 0 else "no"
            instance = Instance(inst_id=i, features=np.random.random(3), label=label)
            data.append((i, instance))
        schema = {"header": ["x0", "x1", "x2"],
                  "sid": "id", "label_name": "y"}
        self.table = session.parallelize(data, include_key=True, partition=8)
        self.table.schema = schema
        self.label_transformer_obj = LabelTransformer()

    def test_get_label_encoder(self):
        self.label_transformer_obj.update_label_encoder(self.table)
        c_label_encoder = {"yes": 1, "no": 0}
        self.assertDictEqual(self.label_transformer_obj.label_encoder, c_label_encoder)

    def test_replace_instance_label(self):
        instance = self.table.first()[1]
        replaced_instance = self.label_transformer_obj.replace_instance_label(instance, self.label_encoder)
        self.assertEqual(replaced_instance.label, self.label_encoder[instance.label])

    def test_transform_data_label(self):
        replaced_data = self.label_transformer_obj.transform_data_label(self.table, self.label_encoder)
        replaced_data.join(self.table, lambda x, y: self.assertEqual(x.label, self.label_encoder[y.label]))

    def test_replace_predict_label(self):
        true_label, predict_label, predict_score, predict_detail, predict_type = 1, 0, 0.1, {
            "1": 0.1, "0": 0.9}, "train"
        predict_result = Instance(inst_id=0,
                                  features=[true_label, predict_label, predict_score, predict_detail, predict_type])
        r_predict_instance = self.label_transformer_obj.replace_predict_label(
            predict_result, self.predict_label_encoder)
        r_predict_result = r_predict_instance.features
        c_predict_result = ["yes", "no", predict_score, {"yes": 0.1, "no": 0.9}, predict_type]
        self.assertEqual(r_predict_result, c_predict_result)

    def tearDown(self):
        session.stop()


if __name__ == '__main__':
    unittest.main()
