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
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.param.feature_selection_param import VarianceOfCoeSelectionParam
from federatedml.protobuf.generated import feature_selection_meta_pb2
import math

LOGGER = log_utils.getLogger()


class VarianceCoeFilter(BaseFilterMethod):
    """
    Filter the columns if coefficient of variance is less than a threshold.
    """
    def __init__(self, filter_param: VarianceOfCoeSelectionParam):
        super().__init__(filter_param)
        self.statics_obj = None

    def _parse_filter_param(self, filter_param):
        self.value_threshold = filter_param.value_threshold

    def set_statics_obj(self, statics_obj):
        self.statics_obj = statics_obj

    def fit(self, data_instances, suffix):
        if self.statics_obj is None:
            self.statics_obj = MultivariateStatisticalSummary(data_instances)

        std_var = self.statics_obj.get_std_variance()
        mean_value = self.statics_obj.get_mean()

        for col_name in self.selection_properties.select_col_names:
            s_v = std_var.get(col_name)
            m_v = mean_value.get(col_name)
            coeff_of_var = math.fabs(s_v / m_v)

            if coeff_of_var >= self.value_threshold:
                self.selection_properties.add_left_col_name(col_name)
            self.selection_properties.add_feature_value(col_name, coeff_of_var)
        self._keep_one_feature(pick_high=True)
        return self

    def get_meta_obj(self, meta_dicts):
        result = feature_selection_meta_pb2.VarianceOfCoeSelectionMeta(value_threshold=self.value_threshold)
        meta_dicts['variance_coe_meta'] = result
        return meta_dicts
