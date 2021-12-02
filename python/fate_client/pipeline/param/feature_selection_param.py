#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

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
from pipeline.param.base_param import BaseParam
from pipeline.param import consts


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

    host_thresholds: List of float or None, default: None
        Set threshold for different host. If None, use same threshold as guest. If provided, the order should map with
        the host id setting.

    """

    def __init__(self, value_threshold=0.0, host_thresholds=None, local_only=False):
        super().__init__()
        self.value_threshold = value_threshold
        self.host_thresholds = host_thresholds
        self.local_only = local_only

    def check(self):
        if not isinstance(self.value_threshold, (float, int)):
            raise ValueError("IV selection param's value_threshold should be float or int")

        if self.host_thresholds is not None:
            if not isinstance(self.host_thresholds, list):
                raise ValueError("IV selection param's host_threshold should be list or None")

        if not isinstance(self.local_only, bool):
            raise ValueError("IV selection param's local_only should be bool")

        return True


class IVPercentileSelectionParam(BaseParam):
    """
    Use information values to select features.

    Parameters
    ----------
    percentile_threshold: float, 0 <= percentile_threshold <= 1.0, default: 1.0
        Percentile threshold for iv_percentile method


    """

    def __init__(self, percentile_threshold=1.0, local_only=False):
        super().__init__()
        self.percentile_threshold = percentile_threshold
        self.local_only = local_only

    def check(self):
        descr = "IV selection param's"
        self.check_decimal_float(self.percentile_threshold, descr)
        self.check_boolean(self.local_only, descr)
        return True


class IVTopKParam(BaseParam):
    """
    Use information values to select features.

    Parameters
    ----------
    k: int, should be greater than 0, default: 10
        Percentile threshold for iv_percentile method
    """

    def __init__(self, k=10, local_only=False):
        super().__init__()
        self.k = k
        self.local_only = local_only

    def check(self):
        descr = "IV selection param's"
        self.check_positive_integer(self.k, descr)
        self.check_boolean(self.local_only, descr)
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


class CommonFilterParam(BaseParam):
    """
    All of the following parameters can set with a single value or a list of those values.
    When setting one single value, it means using only one metric to filter while
    a list represent for using multiple metrics.

    Please note that if some of the following values has been set as list, all of them
    should have same length. Otherwise, error will be raised. And if there exist a list
    type parameter, the metrics should be in list type.

    Parameters
    ----------
    metrics: str or list, default: depends on the specific filter
        Indicate what metrics are used in this filter

    filter_type: str, default: threshold
        Should be one of "threshold", "top_k" or "top_percentile"

    take_high: bool, default: True
        When filtering, taking highest values or not.

    threshold: float or int, default: 1
        If filter type is threshold, this is the threshold value.
        If it is "top_k", this is the k value.
        If it is top_percentile, this is the percentile threshold.

    host_thresholds: List of float or List of List of float or None, default: None
        Set threshold for different host. If None, use same threshold as guest. If provided, the order should map with
        the host id setting.

    select_federated: bool, default: True
        Whether select federated with other parties or based on local variables
    """

    def __init__(self, metrics, filter_type='threshold', take_high=True, threshold=1,
                 host_thresholds=None, select_federated=True):
        super().__init__()
        self.metrics = metrics
        self.filter_type = filter_type
        self.take_high = take_high
        self.threshold = threshold
        self.host_thresholds = host_thresholds
        self.select_federated = select_federated

    def check(self):
        if not isinstance(self.metrics, list):
            for value_name in ["filter_type", "take_high",
                               "threshold", "select_federated"]:
                v = getattr(self, value_name)
                if isinstance(v, list):
                    raise ValueError(f"{value_name}: {v} should not be a list when "
                                     f"metrics: {self.metrics} is not a list")
                setattr(self, value_name, [v])
            setattr(self, "metrics", [self.metrics])
        else:
            expected_length = len(self.metrics)
            for value_name in ["filter_type", "take_high",
                               "threshold", "select_federated"]:
                v = getattr(self, value_name)
                if isinstance(v, list):
                    if len(v) != expected_length:
                        raise ValueError(f"The parameter {v} should have same length "
                                         f"with metrics")
                else:
                    new_v = [v] * expected_length
                    setattr(self, value_name, new_v)

        for v in self.filter_type:
            if v not in ["threshold", "top_k", "top_percentile"]:
                raise ValueError('filter_type should be one of '
                                 '"threshold", "top_k", "top_percentile"')

        descr = "hetero feature selection param's"
        for v in self.take_high:
            self.check_boolean(v, descr)

        for idx, v in enumerate(self.threshold):
            if self.filter_type[idx] == "threshold":
                if not isinstance(v, (float, int)):
                    raise ValueError(descr + f"{v} should be a float or int")
            elif self.filter_type[idx] == 'top_k':
                self.check_positive_integer(v, descr)
            else:
                if not (v == 0 or v == 1):
                    self.check_decimal_float(v, descr)

        if self.host_thresholds is not None:
            if not isinstance(self.host_thresholds, list):
                raise ValueError("IV selection param's host_threshold should be list or None")

        assert isinstance(self.select_federated, list)
        for v in self.select_federated:
            self.check_boolean(v, descr)


class CorrelationFilterParam(BaseParam):
    """
    This filter follow this specific rules:
        1. Sort all the columns from high to low based on specific metric, eg. iv.
        2. Traverse each sorted column. If there exists other columns with whom the
            absolute values of correlation are larger than threshold, they will be filtered.

    Parameters
    ----------
    sort_metric: str, default: iv
        Specify which metric to be used to sort features.

    threshold: float or int, default: 0.1
        Correlation threshold

    select_federated: bool, default: True
        Whether select federated with other parties or based on local variables
    """

    def __init__(self, sort_metric='iv', threshold=0.1, select_federated=True):
        super().__init__()
        self.sort_metric = sort_metric
        self.threshold = threshold
        self.select_federated = select_federated

    def check(self):
        descr = "Correlation Filter param's"

        self.sort_metric = self.sort_metric.lower()
        support_metrics = ['iv']
        if self.sort_metric not in support_metrics:
            raise ValueError(f"sort_metric in Correlation Filter should be one of {support_metrics}")

        self.check_positive_number(self.threshold, descr)


class PercentageValueParam(BaseParam):
    """
    Filter the columns that have a value that exceeds a certain percentage.

    Parameters
    ----------
    upper_pct: float, [0.1, 1.], default: 1.0
        The upper percentage threshold for filtering, upper_pct should not be less than 0.1.

    """

    def __init__(self, upper_pct=1.0):
        super().__init__()
        self.upper_pct = upper_pct

    def check(self):
        descr = "Percentage Filter param's"
        if self.upper_pct not in [0, 1]:
            self.check_decimal_float(self.upper_pct, descr)
        if self.upper_pct < consts.PERCENTAGE_VALUE_LIMIT:
            raise ValueError(descr + f" {self.upper_pct} not supported,"
                                     f" should not be smaller than {consts.PERCENTAGE_VALUE_LIMIT}")
        return True


class ManuallyFilterParam(BaseParam):
    """
    Specified columns that need to be filtered. If exist, it will be filtered directly, otherwise, ignore it.

    Parameters
    ----------
    filter_out_indexes: list of int, default: None
        Specify columns' indexes to be filtered out

    filter_out_names : list of string, default: None
        Specify columns' names to be filtered out

    left_col_indexes: list of int, default: None
        Specify left_col_index

    left_col_names: list of string, default: None
        Specify left col names

    Both Filter_out or left parameters only works for this specific filter. For instances, if you set some columns left
    in this filter but those columns are filtered by other filters, those columns will NOT left in final.

    Please note that (left_col_indexes & left_col_names) cannot use with
        (filter_out_indexes & filter_out_names) simultaneously.

    """

    def __init__(self, filter_out_indexes=None, filter_out_names=None, left_col_indexes=None,
                 left_col_names=None):
        super().__init__()
        self.filter_out_indexes = filter_out_indexes
        self.filter_out_names = filter_out_names
        self.left_col_indexes = left_col_indexes
        self.left_col_names = left_col_names

    def check(self):
        descr = "Manually Filter param's"
        self.check_defined_type(self.filter_out_indexes, descr, ['list', 'NoneType'])
        self.check_defined_type(self.filter_out_names, descr, ['list', 'NoneType'])
        self.check_defined_type(self.left_col_indexes, descr, ['list', 'NoneType'])
        self.check_defined_type(self.left_col_names, descr, ['list', 'NoneType'])

        if (self.filter_out_indexes or self.filter_out_names) is not None and \
                (self.left_col_names or self.left_col_indexes) is not None:
            raise ValueError("(left_col_indexes & left_col_names) cannot use with"
                             " (filter_out_indexes & filter_out_names) simultaneously")
        return True


class FeatureSelectionParam(BaseParam):
    """
    Define the feature selection parameters.

    Parameters
    ----------
    select_col_indexes: list or int, default: -1
        Specify which columns need to calculated. -1 represent for all columns.

    select_names : list of string, default: []
        Specify which columns need to calculated. Each element in the list represent for a column name in header.

    filter_methods: list, ["manually", "iv_filter", "statistic_filter",
                            "psi_filter", “hetero_sbt_filter", "homo_sbt_filter",
                             "hetero_fast_sbt_filter", "percentage_value",
                             "vif_filter", "correlation_filter"],
                 default: ["manually"]

        The following methods will be deprecated in future version:
        "unique_value", "iv_value_thres", "iv_percentile",
        "coefficient_of_variation_value_thres", "outlier_cols"

        Specify the filter methods used in feature selection. The orders of filter used is depended on this list.
        Please be notified that, if a percentile method is used after some certain filter method,
        the percentile represent for the ratio of rest features.

        e.g. If you have 10 features at the beginning. After first filter method, you have 8 rest. Then, you want
        top 80% highest iv feature. Here, we will choose floor(0.8 * 8) = 6 features instead of 8.

    unique_param: filter the columns if all values in this feature is the same

    iv_value_param: Use information value to filter columns. If this method is set, a float threshold need to be provided.
        Filter those columns whose iv is smaller than threshold. Will be deprecated in the future.

    iv_percentile_param: Use information value to filter columns. If this method is set, a float ratio threshold
        need to be provided. Pick floor(ratio * feature_num) features with higher iv. If multiple features around
        the threshold are same, all those columns will be keep. Will be deprecated in the future.

    variance_coe_param: Use coefficient of variation to judge whether filtered or not.
        Will be deprecated in the future.

    outlier_param: Filter columns whose certain percentile value is larger than a threshold.
        Will be deprecated in the future.

    percentage_value_param: Filter the columns that have a value that exceeds a certain percentage.

    iv_param: Setting how to filter base on iv. It support take high mode only. All of "threshold",
        "top_k" and "top_percentile" are accepted. Check more details in CommonFilterParam. To
        use this filter, hetero-feature-binning module has to be provided.

    statistic_param: Setting how to filter base on statistic values. All of "threshold",
        "top_k" and "top_percentile" are accepted. Check more details in CommonFilterParam.
        To use this filter, data_statistic module has to be provided.

    psi_param: Setting how to filter base on psi values. All of "threshold",
        "top_k" and "top_percentile" are accepted. Its take_high properties should be False
        to choose lower psi features. Check more details in CommonFilterParam.
        To use this filter, data_statistic module has to be provided.

    need_run: bool, default True
        Indicate if this module needed to be run

    """

    def __init__(self, select_col_indexes=-1, select_names=None, filter_methods=None,
                 unique_param=UniqueValueParam(),
                 iv_value_param=IVValueSelectionParam(),
                 iv_percentile_param=IVPercentileSelectionParam(),
                 iv_top_k_param=IVTopKParam(),
                 variance_coe_param=VarianceOfCoeSelectionParam(),
                 outlier_param=OutlierColsSelectionParam(),
                 manually_param=ManuallyFilterParam(),
                 percentage_value_param=PercentageValueParam(),
                 iv_param=CommonFilterParam(metrics=consts.IV),
                 statistic_param=CommonFilterParam(metrics=consts.MEAN),
                 psi_param=CommonFilterParam(metrics=consts.PSI,
                                             take_high=False),
                 vif_param=CommonFilterParam(metrics=consts.VIF,
                                             threshold=5.0,
                                             take_high=False),
                 sbt_param=CommonFilterParam(metrics=consts.FEATURE_IMPORTANCE),
                 correlation_param=CorrelationFilterParam(),
                 need_run=True
                 ):
        super(FeatureSelectionParam, self).__init__()
        self.correlation_param = correlation_param
        self.vif_param = vif_param
        self.select_col_indexes = select_col_indexes
        if select_names is None:
            self.select_names = []
        else:
            self.select_names = select_names
        if filter_methods is None:
            self.filter_methods = [consts.MANUALLY_FILTER]
        else:
            self.filter_methods = filter_methods

        # deprecate in the future
        self.unique_param = copy.deepcopy(unique_param)
        self.iv_value_param = copy.deepcopy(iv_value_param)
        self.iv_percentile_param = copy.deepcopy(iv_percentile_param)
        self.iv_top_k_param = copy.deepcopy(iv_top_k_param)
        self.variance_coe_param = copy.deepcopy(variance_coe_param)
        self.outlier_param = copy.deepcopy(outlier_param)
        self.percentage_value_param = copy.deepcopy(percentage_value_param)

        self.manually_param = copy.deepcopy(manually_param)
        self.iv_param = copy.deepcopy(iv_param)
        self.statistic_param = copy.deepcopy(statistic_param)
        self.psi_param = copy.deepcopy(psi_param)
        self.sbt_param = copy.deepcopy(sbt_param)
        self.need_run = need_run

    def check(self):
        descr = "hetero feature selection param's"

        self.check_defined_type(self.filter_methods, descr, ['list'])

        for idx, method in enumerate(self.filter_methods):
            method = method.lower()
            self.check_valid_value(method, descr, [consts.UNIQUE_VALUE, consts.IV_VALUE_THRES, consts.IV_PERCENTILE,
                                                   consts.COEFFICIENT_OF_VARIATION_VALUE_THRES, consts.OUTLIER_COLS,
                                                   consts.MANUALLY_FILTER, consts.PERCENTAGE_VALUE,
                                                   consts.IV_FILTER, consts.STATISTIC_FILTER, consts.IV_TOP_K,
                                                   consts.PSI_FILTER, consts.HETERO_SBT_FILTER,
                                                   consts.HOMO_SBT_FILTER, consts.HETERO_FAST_SBT_FILTER,
                                                   consts.VIF_FILTER, consts.CORRELATION_FILTER])

            self.filter_methods[idx] = method

        self.check_defined_type(self.select_col_indexes, descr, ['list', 'int'])

        self.unique_param.check()
        self.iv_value_param.check()
        self.iv_percentile_param.check()
        self.iv_top_k_param.check()
        self.variance_coe_param.check()
        self.outlier_param.check()
        self.manually_param.check()
        self.percentage_value_param.check()

        self.iv_param.check()
        for th in self.iv_param.take_high:
            if not th:
                raise ValueError("Iv filter should take higher iv features")
        for m in self.iv_param.metrics:
            if m != consts.IV:
                raise ValueError("For iv filter, metrics should be 'iv'")

        self.statistic_param.check()
        self.psi_param.check()
        for th in self.psi_param.take_high:
            if th:
                raise ValueError("PSI filter should take lower psi features")
        for m in self.psi_param.metrics:
            if m != consts.PSI:
                raise ValueError("For psi filter, metrics should be 'psi'")

        self.sbt_param.check()
        for th in self.sbt_param.take_high:
            if not th:
                raise ValueError("SBT filter should take higher feature_importance features")
        for m in self.sbt_param.metrics:
            if m != consts.FEATURE_IMPORTANCE:
                raise ValueError("For SBT filter, metrics should be 'feature_importance'")

        self.vif_param.check()
        for m in self.vif_param.metrics:
            if m != consts.VIF:
                raise ValueError("For VIF filter, metrics should be 'vif'")

        self.correlation_param.check()
