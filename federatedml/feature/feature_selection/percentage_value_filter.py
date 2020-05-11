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

from arch.api.utils import log_utils
from federatedml.feature.feature_selection.filter_base import BaseFilterMethod
from federatedml.param.feature_selection_param import PercentageValueParam
from federatedml.protobuf.generated import feature_selection_meta_pb2
from federatedml.statistic.statics import MultivariateStatisticalSummary

LOGGER = log_utils.getLogger()


class PercentageValueFilter(BaseFilterMethod):
    """
    filter the columns if all values in this feature is the same

    """

    def __init__(self, filter_param: PercentageValueParam):
        super().__init__(filter_param)
        self.statics_obj = None

    def _parse_filter_param(self, filter_param):
        self.upper_pct = filter_param.upper_pct
        self.error = filter_param.error

    def set_statics_obj(self, statics_obj):
        self.statics_obj = statics_obj

    def fit(self, data_instances, suffix):
        if self.statics_obj is None:
            select_col_idx = self.selection_properties.select_col_indexes
            self.statics_obj = MultivariateStatisticalSummary(data_instances, cols_index=select_col_idx)
        quantile_dict = {col_name: [] for col_name in self.selection_properties.select_col_names}
        pct_inteval = self.upper_pct / 2
        i = 0
        while i * pct_inteval <= 1:
            pct_res = self.statics_obj.get_quantile_point(i * pct_inteval, error=self.error)
            for col_name, pct_list in quantile_dict.items():
                pct_list.append(pct_res[col_name])
            i += 1

        for col_name in self.selection_properties.select_col_names:
            pct_list = quantile_dict.get(col_name)
            if len(pct_list) == len(set(pct_list)):
                self.selection_properties.add_left_col_name(col_name)
                self.selection_properties.add_feature_value(col_name, False)
            else:
                self.selection_properties.add_feature_value(col_name, True)

        self._keep_one_feature(pick_high=True)
        return self

    def get_meta_obj(self, meta_dicts):
        result = feature_selection_meta_pb2.PercentageValueFilterMeta(upper_pct=self.upper_pct,
                                                                      error=self.error)
        meta_dicts['pencentage_value_meta'] = result
        return meta_dicts
