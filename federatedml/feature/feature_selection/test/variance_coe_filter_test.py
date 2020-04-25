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

import unittest

from arch.api import session
import numpy as np
from federatedml.feature.feature_selection.filter_factory import get_filter
from federatedml.param.feature_selection_param import FeatureSelectionParam
from federatedml.util import consts
from federatedml.feature.feature_selection.selection_properties import SelectionProperties
import uuid


class TestVarianceCoeFilter(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        session.init(self.job_id)

    def gen_data(self, data_num, feature_num, partition):
        data = []
        header = [str(i) for i in range(feature_num)]
        # col_2 = np.random.rand(data_num)
        col_data = []
        for _ in range(feature_num - 1):
            while True:
                col_1 = np.random.rand(data_num)
                if np.mean(col_1) != 0:
                    break
            col_data.append(col_1)
        col_data.append(10 * np.ones(data_num))

        for key in range(data_num):
            data.append((key, np.array([col[key] for col in col_data])))

        result = session.parallelize(data, include_key=True, partition=partition)
        result.schema = {'header': header}
        self.header = header

        self.coe_list = []
        for col in col_data:
            self.coe_list.append(np.std(col) / np.mean(col))
        return result

    def test_unique_logic(self):
        data_table = self.gen_data(1000, 10, 48)
        select_param = FeatureSelectionParam()
        select_param.variance_coe_param.value_threshold = 0.1
        filter_obj = get_filter(consts.COEFFICIENT_OF_VARIATION_VALUE_THRES, select_param)
        select_properties = SelectionProperties()
        select_properties.set_header(self.header)
        select_properties.set_last_left_col_indexes([x for x in range(len(self.header))])
        select_properties.set_select_all_cols()
        filter_obj.set_selection_properties(select_properties)
        res_select_properties = filter_obj.fit(data_table, suffix='').selection_properties
        result = [self.header[idx] for idx, x in enumerate(self.coe_list)
                  if x >= select_param.variance_coe_param.value_threshold]

        self.assertEqual(res_select_properties.all_left_col_names, result)
        self.assertEqual(len(res_select_properties.all_left_col_names), 9)
        data_table.destroy()

    def tearDown(self):
        session.stop()
        try:
            session.cleanup("*", self.job_id, True)
        except EnvironmentError:
            pass
        try:
            session.cleanup("*", self.job_id, False)
        except EnvironmentError:
            pass


if __name__ == '__main__':
    unittest.main()