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
#
from federatedml.param.base_param import BaseParam
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.util import consts
import copy


class UniqueValueParam(BaseParam):
    """
    Use the difference between max-value and min-value to judge.

    Parameters
    ----------
    eps: float, default: 1e-5
        The column(s) will be filtered if its difference is smaller than eps.
    """

    def __init__(self, eps=1e-5):
        self.eps = eps

    def check(self):
        descr = "Unique value param's"
        self.check_positive_number(self.eps, descr)
        return True


class IVValueSelectionParam(BaseParam):
    """
    Use information values to select features.

    Parameters
    ----------
    value_threshold: float, default: 1.0
        Used if iv_value_thres method is used in feature selection.

    """

    def __init__(self, value_threshold=0.0):
        self.value_threshold = value_threshold

    def check(self):
        if not isinstance(self.value_threshold, (float, int)):
            raise ValueError("IV selection param's value_threshold should be float or int")

        return True


class IVPercentileSelectionParam(BaseParam):
    """
    Use information values to select features.

    Parameters
    ----------
    percentile_threshold: float, 0 <= percentile_threshold <= 1.0, default: 1.0
        Percentile threshold for iv_percentile method


    """

    def __init__(self, percentile_threshold=1.0):
        self.percentile_threshold = percentile_threshold

    def check(self):
        descr = "IV selection param's"
        self.check_decimal_float(self.percentile_threshold, descr)
        return True


class VarianceOfCoeSelectionParam(BaseParam):
    """
    Use coefficient of variation to select features. When judging, the absolute value will be used.

    Parameters
    ----------
    value_threshold: float, default: 1.0
        Used if coefficient_of_variation_value_thres method is used in feature selection. Filter those
        columns who has smaller coefficient of variance than the threshold.

    """

    def __init__(self, value_threshold=1.0):
        self.value_threshold = value_threshold

    def check(self):
        descr = "Coff of Variances param's"
        self.check_positive_number(self.value_threshold, descr)
        return True


class OutlierColsSelectionParam(BaseParam):
    """
    Given percentile and threshold. Judge if this quantile point is larger than threshold. Filter those larger ones.

    Parameters
    ----------
    percentile: float, [0., 1.] default: 1.0
        The percentile points to compare.

    upper_threshold: float, default: 1.0
        Percentile threshold for coefficient_of_variation_percentile method

    """

    def __init__(self, percentile=1.0, upper_threshold=1.0):
        self.percentile = percentile
        self.upper_threshold = upper_threshold

    def check(self):
        descr = "Outlier Filter param's"
        self.check_decimal_float(self.percentile, descr)
        self.check_defined_type(self.upper_threshold, descr, ['float', 'int'])
        return True


class FeatureSelectionParam(BaseParam):
    """
    Define the feature selection parameters.

    Parameters
    ----------
    select_cols: list or int, default: -1
        Specify which columns need to calculated. -1 represent for all columns.

    filter_methods: list, ["unique_value", "iv_value_thres", "iv_percentile",
                "coefficient_of_variation_value_thres", "outlier_cols"],
                 default: ["unique_value"]

        Specify the filter methods used in feature selection. The orders of filter used is depended on this list.
        Please be notified that, if a percentile method is used after some certain filter method,
        the percentile represent for the ratio of rest features.

        e.g. If you have 10 features at the beginning. After first filter method, you have 8 rest. Then, you want
        top 80% highest iv feature. Here, we will choose floor(0.8 * 8) = 6 features instead of 8.

    unique_value: filter the columns if all values in this feature is the same

    iv_value_thres: Use information value to filter columns. If this method is set, a float threshold need to be provided.
        Filter those columns whose iv is smaller than threshold.

    iv_percentile: Use information value to filter columns. If this method is set, a float ratio threshold
        need to be provided. Pick floor(ratio * feature_num) features with higher iv. If multiple features around
        the threshold are same, all those columns will be keep.

    coefficient_of_variation_value_thres: Use coefficient of variation to judge whether filtered or not.

    outlier_cols: Filter columns whose certain percentile value is larger than a threshold.

    need_run: bool, default True
        Indicate if this module needed to be run

    """

    def __init__(self, select_cols=-1, filter_methods=None, local_only=False,
                 unique_param=UniqueValueParam(),
                 iv_value_param=IVValueSelectionParam(),
                 iv_percentile_param=IVPercentileSelectionParam(),
                 variance_coe_param=VarianceOfCoeSelectionParam(),
                 outlier_param=OutlierColsSelectionParam(),
                 need_run=True
                 ):
        super(FeatureSelectionParam, self).__init__()
        self.select_cols = select_cols
        if filter_methods is None:
            self.filter_methods = [consts.UNIQUE_VALUE]
        else:
            self.filter_methods = filter_methods

        self.local_only = local_only
        self.unique_param = copy.deepcopy(unique_param)
        self.iv_value_param = copy.deepcopy(iv_value_param)
        self.iv_percentile_param = copy.deepcopy(iv_percentile_param)
        self.variance_coe_param = copy.deepcopy(variance_coe_param)
        self.outlier_param = copy.deepcopy(outlier_param)
        self.need_run = need_run

    def check(self):
        descr = "hetero feature selection param's"

        self.check_defined_type(self.filter_methods, descr, ['list'])

        for idx, method in enumerate(self.filter_methods):
            method = method.lower()
            self.check_valid_value(method, descr, ["unique_value", "iv_value_thres", "iv_percentile",
                                              "coefficient_of_variation_value_thres",
                                              "outlier_cols"])
            self.filter_methods[idx] = method

        self.check_defined_type(self.select_cols, descr, ['list', 'int'])

        self.check_boolean(self.local_only, descr)
        self.unique_param.check()
        self.iv_value_param.check()
        self.iv_percentile_param.check()
        self.variance_coe_param.check()
        self.outlier_param.check()

