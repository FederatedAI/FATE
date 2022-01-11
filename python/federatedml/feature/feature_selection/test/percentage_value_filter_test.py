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

from fate_arch.session import computing_session as session
import numpy as np
from federatedml.feature.feature_selection.filter_factory import get_filter
from federatedml.param.feature_selection_param import FeatureSelectionParam
from federatedml.util import consts
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.feature.feature_selection.selection_properties import SelectionProperties
import uuid
import random


class TestPercentageValueFilter(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        session.init(self.job_id)

    def gen_data(self, data_num, partition):
        col_data = []
        header = [str(i) for i in range(6)]
        mode_num = int(0.8 * data_num)
        other_num = data_num - mode_num
        col_1 = np.array([1] * mode_num + [0] * other_num)
        random.shuffle(col_1)
        col_data.append(col_1)

        mode_num = int(0.799 * data_num)
        other_num = data_num - mode_num
        col_1 = np.array([1] * mode_num + [0] * other_num)
        random.shuffle(col_1)
        col_data.append(col_1)

        mode_num = int(0.801 * data_num)
        other_num = data_num - mode_num
        col_1 = np.array([1] * mode_num + [0] * other_num)
        random.shuffle(col_1)
        col_data.append(col_1)

        col_2 = np.random.randn(data_num)
        col_data.append(col_2)

        mode_num = int(0.2 * data_num)
        other_num = data_num - mode_num
        col_1 = np.array([0.5] * mode_num + list(np.random.randn(other_num)))
        print("col 0.5 count: {}".format(list(col_1).count(0.5)))
        random.shuffle(col_1)
        col_data.append(col_1)

        mode_num = int(0.79 * data_num)
        other_num = data_num - mode_num
        col_1 = np.array([0.5] * mode_num + list(np.random.randn(other_num)))
        random.shuffle(col_1)
        col_data.append(col_1)

        data = []
        data_2 = []
        for key in range(data_num):
            features = np.array([col[key] for col in col_data])
            inst = Instance(inst_id=key, features=features, label=key % 2)
            data.append((key, inst))

            sparse_vec = SparseVector(indices=[i for i in range(len(features))], data=features, shape=len(features))
            inst_2 = Instance(inst_id=key, features=sparse_vec, label=key % 2)
            data_2.append((key, inst_2))

        result = session.parallelize(data, include_key=True, partition=partition)
        result_2 = session.parallelize(data_2, include_key=True, partition=partition)
        result.schema = {'header': header}
        result_2.schema = {'header': header}

        self.header = header
        return result, result_2

    def test_percentage_value_logic(self):
        data_table, data_table_2 = self.gen_data(1000, 48)
        self._run_filter(data_table)
        self._run_filter(data_table_2)

    def _run_filter(self, data_table):
        select_param = FeatureSelectionParam()
        select_param.percentage_value_param.upper_pct = 0.2
        filter_obj = get_filter(consts.PERCENTAGE_VALUE, select_param)
        select_properties = SelectionProperties()
        select_properties.set_header(self.header)
        select_properties.set_last_left_col_indexes([x for x in range(len(self.header))])
        select_properties.set_select_all_cols()
        filter_obj.set_selection_properties(select_properties)
        res_select_properties = filter_obj.fit(data_table, suffix='').selection_properties
        left_cols = [3, 4]
        self.assertEqual(res_select_properties.all_left_col_names, [self.header[x] for x in left_cols])

    def tearDown(self):
        session.stop()


if __name__ == '__main__':
    unittest.main()
