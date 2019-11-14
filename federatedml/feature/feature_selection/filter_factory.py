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

from federatedml.feature.feature_selection.unique_value_filter import UniqueValueFilter
from federatedml.feature.feature_selection import iv_value_select_filter, iv_percentile_filter
from federatedml.feature.feature_selection.variance_coe_filter import VarianceCoeFilter
from federatedml.feature.feature_selection.outlier_filter import OutlierFilter
from federatedml.feature.feature_selection.manually_filter import ManuallyFilter
from federatedml.param.feature_selection_param import FeatureSelectionParam
from federatedml.util import consts


def get_filter(filter_name, model_param: FeatureSelectionParam, role=consts.GUEST):
    if filter_name == consts.UNIQUE_VALUE:
        unique_param = model_param.unique_param
        return UniqueValueFilter(unique_param)

    elif filter_name == consts.IV_VALUE_THRES:
        iv_param = model_param.iv_value_param
        if role == consts.GUEST:
            return iv_value_select_filter.Guest(iv_param)
        else:
            return iv_value_select_filter.Host(iv_param)

    elif filter_name == consts.IV_PERCENTILE:
        iv_param = model_param.iv_percentile_param
        if role == consts.GUEST:
            return iv_percentile_filter.Guest(iv_param)
        else:
            return iv_percentile_filter.Host(iv_param)

    elif filter_name == consts.COEFFICIENT_OF_VARIATION_VALUE_THRES:
        coe_param = model_param.variance_coe_param
        return VarianceCoeFilter(coe_param)

    elif filter_name == consts.OUTLIER_COLS:
        outlier_param = model_param.outlier_param
        return OutlierFilter(outlier_param)

    elif filter_name == consts.MANUALLY_FILTER:
        manually_param = model_param.manually_param
        return ManuallyFilter(manually_param)

    else:
        raise ValueError("filter method: {} does not exist".format(filter_name))
