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


class TestUniqueValueFilter(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        session.init(self.job_id)

    def gen_data(self, data_num, partition):
        data = []
        header = [str(i) for i in range(2)]
        col_1 = np.random.randint(100) * np.ones(data_num)
        col_2 = np.random.randn(data_num)
        for key in range(data_num):
            data.append((key, np.array([col_1[key], col_2[key]])))

        result = session.parallelize(data, include_key=True, partition=partition)
        result.schema = {'header': header}
        self.header = header
        return result

    def test_unique_logic(self):
        data_table = self.gen_data(1000, 48)
        select_param = FeatureSelectionParam()
        filter_obj = get_filter(consts.UNIQUE_VALUE, select_param)
        select_properties = SelectionProperties()
        select_properties.set_header(self.header)
        select_properties.set_last_left_col_indexes([x for x in range(len(self.header))])
        select_properties.set_select_all_cols()
        filter_obj.set_selection_properties(select_properties)
        res_select_properties = filter_obj.fit(data_table, suffix='').selection_properties
        self.assertEqual(res_select_properties.all_left_col_names, [self.header[1]])
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