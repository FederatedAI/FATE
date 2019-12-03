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
from federatedml.param.feature_selection_param import OutlierColsSelectionParam
from federatedml.protobuf.generated import feature_selection_meta_pb2

LOGGER = log_utils.getLogger()


class OutlierFilter(BaseFilterMethod):
    """
    Filter the columns if coefficient of variance is less than a threshold.
    """
    def __init__(self, filter_param: OutlierColsSelectionParam):
        super().__init__(filter_param)
        self.statics_obj = None

    def _parse_filter_param(self, filter_param: OutlierColsSelectionParam):
        self.percentile = filter_param.percentile
        self.upper_threshold = filter_param.upper_threshold

    def set_statics_obj(self, statics_obj):
        self.statics_obj = statics_obj

    def fit(self, data_instances, suffix):
        if self.statics_obj is None:
            self.statics_obj = MultivariateStatisticalSummary(data_instances)

        quantile_points = self.statics_obj.get_quantile_point(self.percentile)

        for col_name in self.selection_properties.select_col_names:
            quantile_value = quantile_points.get(col_name)
            if quantile_value < self.upper_threshold:
                self.selection_properties.add_left_col_name(col_name)
            self.selection_properties.add_feature_value(col_name, quantile_value)
        self._keep_one_feature(pick_high=True)
        return self

    def get_meta_obj(self, meta_dicts):
        result = feature_selection_meta_pb2.OutlierColsSelectionMeta(percentile=self.percentile,
                                                                     upper_threshold=self.upper_threshold)
        meta_dicts['outlier_meta'] = result
        return meta_dicts
