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
import uuid

import numpy as np

from fate_arch.session import computing_session as session
from federatedml.feature.feature_selection.filter_factory import get_filter
from federatedml.feature.feature_selection.model_adapter.adapter_factory import adapter_factory
from federatedml.feature.feature_selection.selection_properties import SelectionProperties
from federatedml.feature.hetero_feature_selection.base_feature_selection import BaseHeteroFeatureSelection
from federatedml.param.feature_selection_param import FeatureSelectionParam
from federatedml.param.statistics_param import StatisticsParam
from federatedml.statistic.data_statistics import DataStatistics
from federatedml.util import consts
from federatedml.feature.instance import Instance


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
            data.append((key, Instance(features=np.array([col_1[key], col_2[key]]))))

        result = session.parallelize(data, include_key=True, partition=partition)
        result.schema = {'header': header}
        self.header = header
        return result

    def test_unique_logic(self):
        data_table = self.gen_data(1000, 48)
        select_param = FeatureSelectionParam()
        selection_obj = self._make_selection_obj(data_table)

        filter_obj = get_filter(consts.UNIQUE_VALUE, select_param, model=selection_obj)
        select_properties = SelectionProperties()
        select_properties.set_header(self.header)
        select_properties.set_last_left_col_indexes([x for x in range(len(self.header))])
        select_properties.set_select_all_cols()
        filter_obj.set_selection_properties(select_properties)
        res_select_properties = filter_obj.fit(data_table, suffix='').selection_properties
        self.assertEqual(res_select_properties.all_left_col_names, [self.header[1]])

    def _make_selection_obj(self, data_table):
        statistics_param = StatisticsParam(statistics="summary")
        statistics_param.check()
        print(statistics_param.statistics)
        test_obj = DataStatistics()

        test_obj.model_param = statistics_param
        test_obj._init_model(statistics_param)
        test_obj.fit(data_table)

        adapter = adapter_factory(consts.STATISTIC_MODEL)
        meta_obj = test_obj.export_model()['StatisticMeta']
        param_obj = test_obj.export_model()['StatisticParam']

        iso_model = adapter.convert(meta_obj, param_obj)
        selection_obj = BaseHeteroFeatureSelection()
        selection_obj.isometric_models = {consts.STATISTIC_MODEL: iso_model}
        return selection_obj

    def tearDown(self):
        session.stop()


if __name__ == '__main__':
    unittest.main()
