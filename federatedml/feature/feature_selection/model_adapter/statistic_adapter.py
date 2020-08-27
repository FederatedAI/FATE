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

from federatedml.feature.feature_selection.model_adapter import isometric_model
from federatedml.feature.feature_selection.model_adapter.adapter_base import BaseAdapter


class StatisticAdapter(BaseAdapter):

    def convert(self, model_meta, model_param):
        result = isometric_model.IsometricModel()
        self_values = model_param.self_values
        for value_obj in list(self_values.results):
            metric_name = value_obj.value_name
            values = list(value_obj.values)
            col_names = list(value_obj.col_names)
            if len(values) != len(col_names):
                raise ValueError(f"The length of values are not equal to the length"
                                 f" of col_names with metric_name: {metric_name}")
            metric_info = isometric_model.SingleMetricInfo(values, col_names)

            result.add_metric_value(metric_name, metric_info)
        return result
