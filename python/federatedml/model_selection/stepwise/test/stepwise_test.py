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
import uuid

from fate_arch.common import profile
from fate_arch.session import Session
from federatedml.feature.instance import Instance
from federatedml.model_selection.stepwise.hetero_stepwise import HeteroStepwise
from federatedml.util import consts
from federatedml.util import data_io

profile._PROFILE_LOG_ENABLED = False


class TestStepwise(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        self.session = Session.create(0, 0).init_computing(self.job_id).computing
        model = HeteroStepwise()
        model.__setattr__('role', consts.GUEST)
        model.__setattr__('fit_intercept', True)

        self.model = model
        data_num = 100
        feature_num = 5
        bool_list = [True, False, True, True, False]
        self.str_mask = "10110"
        self.header = ["x1", "x2", "x3", "x4", "x5"]
        self.mask = self.prepare_mask(bool_list)
        self.table = self.prepare_data(data_num, feature_num, self.header, "id", "y")

    def prepare_data(self, data_num, feature_num, header, sid_name, label_name):
        final_result = []
        for i in range(data_num):
            tmp = i * np.ones(feature_num)
            inst = Instance(inst_id=i, features=tmp, label=0)
            tmp = (i, inst)
            final_result.append(tmp)
        table = self.session.parallelize(final_result,
                                    include_key=True,
                                    partition=3)
        schema = data_io.make_schema(header, sid_name, label_name)
        table.schema = schema
        return table

    def prepare_mask(self, bool_list):
        mask = np.array(bool_list, dtype=bool)
        return mask

    def test_get_dfe(self):
        real_dfe = 4
        dfe = HeteroStepwise.get_dfe(self.model, self.str_mask)
        self.assertEqual(dfe, real_dfe)

    def test_drop_one(self):
        real_masks = [np.array([0, 0, 1, 1, 0], dtype=bool), np.array([1, 0, 0, 1, 0], dtype=bool),
                      np.array([1, 0, 1, 0, 0], dtype=bool)]
        mask_generator = HeteroStepwise.drop_one(self.mask)
        i = 0
        for mask in mask_generator:
            np.testing.assert_array_equal(mask, real_masks[i],
                                          f"In stepwise_test drop one: mask{mask} not equal to expected {real_masks[i]}")
            i += 1

    def test_add_one(self):
        real_masks = [np.array([1, 1, 1, 1, 0], dtype=bool), np.array([1, 0, 1, 1, 1], dtype=bool)]
        mask_generator = HeteroStepwise.add_one(self.mask)
        i = 0
        for mask in mask_generator:
            np.testing.assert_array_equal(mask, real_masks[i],
                                          f"In stepwise_test add one: mask{mask} not equal to expected {real_masks[i]}")
            i += 1

    def test_mask2string(self):
        real_str_mask = "1011010110"
        str_mask = HeteroStepwise.mask2string(self.mask, self.mask)
        self.assertTrue(str_mask == real_str_mask)

    def test_string2mask(self):
        real_mask = np.array([1, 0, 1, 1, 0], dtype=bool)
        mask = HeteroStepwise.string2mask(self.str_mask)
        np.testing.assert_array_equal(mask, real_mask)

    def test_get_to_enter(self):
        real_to_enter = ["x2", "x5"]
        to_enter = self.model.get_to_enter(self.mask, self.mask, self.header)
        self.assertListEqual(to_enter, real_to_enter)

    def tearDown(self):
        self.session.stop()
        try:
            self.session.cleanup("*", self.job_id)
        except EnvironmentError:
            pass


if __name__ == '__main__':
    unittest.main()