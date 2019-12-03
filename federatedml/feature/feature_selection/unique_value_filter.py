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

from federatedml.feature.feature_selection.filter_base import BaseFilterMethod
from federatedml.param.feature_selection_param import UniqueValueParam
from federatedml.protobuf.generated import feature_selection_meta_pb2
from federatedml.statistic.statics import MultivariateStatisticalSummary
import math


class UniqueValueFilter(BaseFilterMethod):
    """
    filter the columns if all values in this feature is the same

    """
    def __init__(self, filter_param: UniqueValueParam):
        super().__init__(filter_param)
        self.statics_obj = None

    def _parse_filter_param(self, filter_param):
        self.eps = filter_param.eps

    def set_statics_obj(self, statics_obj):
        self.statics_obj = statics_obj

    def fit(self, data_instances, suffix):
        if self.statics_obj is None:
            self.statics_obj = MultivariateStatisticalSummary(data_instances)

        max_values = self.statics_obj.get_max()
        min_values = self.statics_obj.get_min()

        for col_name in self.selection_properties.select_col_names:
            min_max_diff = math.fabs(max_values[col_name] - min_values[col_name])
            if min_max_diff >= self.eps:
                self.selection_properties.add_left_col_name(col_name)
            self.selection_properties.add_feature_value(col_name, min_max_diff)
        self._keep_one_feature(pick_high=True)
        return self

    def get_meta_obj(self, meta_dicts):
        result = feature_selection_meta_pb2.UniqueValueMeta(eps=self.eps)
        meta_dicts['unique_meta'] = result
        return meta_dicts


