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

import math
import operator
import random

from arch.api.proto import feature_selection_meta_pb2
from arch.api.proto import feature_selection_param_pb2
from arch.api.utils import log_utils
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.param.param import FeatureBinningParam, UniqueValueParam
from federatedml.statistic.data_overview import get_header
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class FilterMethod(object):
    def __init__(self):
        self.feature_values = {}
        self.host_feature_values = {}
        self.cols = []
        self.left_cols = {}
        self.host_cols = {}
        self.cols_dict = {}

    def fit(self, data_instances):
        """
        Filter data_instances for the specified columns

        Parameters
        ----------
        data_instances : DTable,
            Input data

        Returns
        -------
        A list of index of columns left.

        """
        pass

    def display_feature_result(self, party_name='Base'):
        class_name = self.__class__.__name__
        for col_name, feature_value in self.feature_values.items():
            LOGGER.info("[Result][FeatureSelection][{}], in {}, col: {} 's feature value is {}".format(
                party_name, class_name, col_name, feature_value
            ))


    def _keep_one_feature(self, pick_high=True, left_cols=None, feature_values=None):
        """
        Make sure at least one feature can be left after filtering.

        Parameters
        ----------
        pick_high: bool
            Set when none of value left, choose the highest one or lowest one. True means highest one while
            False mease lowest one.

        Returns
        -------
        A list of index of columns left.
        """
        if left_cols is None:
            left_cols = self.left_cols

        if feature_values is None:
            feature_values = self.feature_values

        for col_name, is_left in left_cols.items():
            if is_left:
                return left_cols

        # random pick one
        if len(feature_values) == 0:
            left_key = random.choice(left_cols.keys())
        # pick the column with highest value
        else:
            result = sorted(left_cols.items(), key=operator.itemgetter(1), reverse=pick_high)
            left_key = result[0][0]

        left_cols[left_key] = True
        return left_cols

    def get_meta_obj(self):
        pass

    def get_param_obj(self):
        pass

    def _init_cols(self, data_instances):
        header = get_header(data_instances)
        if self.cols == -1:
            self.cols = header

        for col in self.cols:
            col_index = header.index(col)
            self.cols_dict[col] = col_index

    @staticmethod
    def filter_one_party(party_variances, pick_high, value_threshold):
        left_cols = {}
        for col_name, var_value in party_variances.items():
            if pick_high:
                if var_value > value_threshold:
                    left_cols[col_name] = True
                else:
                    left_cols[col_name] = False
            else:
                if var_value < value_threshold:
                    left_cols[col_name] = True
                else:
                    left_cols[col_name] = False
        return left_cols


class UnionPercentileFilter(FilterMethod):
    """
    Use for all union percentile filter methods

    Parameters
    ----------
    local_variance : list,
        The variance of guest party

    host_variances : list,
        The variance of guest party

    percentile : float
        The threshold percentile

    """

    def __init__(self, local_variance, host_variances, percentile, pick_high=True):
        super(UnionPercentileFilter, self).__init__()
        self.local_variance = local_variance  # dict
        self.host_variances = host_variances  # dict of dict
        self.percentiles = percentile
        self.value_threshold = 0.0
        self.pick_high = pick_high

    def fit(self, data_instances=None):
        """
        Fit local variances and each host vaiances

        Parameters
        ----------
        data_instances : Useless, exist for extension
        """
        self.get_value_threshold()
        local_left_cols = self.filter_one_party(self.local_variance, self.pick_high, self.value_threshold)
        local_left_cols = self.keep_one(self.local_variance, local_left_cols)
        host_left_cols = {}
        for host_name, host_vaiances in self.host_variances.items():
            left_col = self.filter_one_party(host_vaiances, self.pick_high, self.value_threshold)
            left_col = self.keep_one(host_vaiances, left_col)
            host_left_cols[host_name] = left_col
        return local_left_cols, host_left_cols

    def keep_one(self, variances, left_cols):
        if len(variances) == 0:
            return left_cols

        for col_name, is_left in left_cols.items():
            if is_left:
                return left_cols

        if self.pick_high:
            max_value = float('-inf')
            max_key = None
            for col_name, feature_value in variances.items():
                if feature_value > max_value:
                    max_value = feature_value
                    max_key = col_name
            left_cols[max_key] = True
        else:
            min_value = float('inf')
            min_key = None
            for col_name, feature_value in variances.items():
                if feature_value < min_value:
                    min_value = feature_value
                    min_key = col_name
            left_cols[min_key] = True
        return left_cols

    def get_value_threshold(self):
        total_values = []
        total_values.extend(list(self.local_variance.values()))

        for h_v in self.host_variances.values():
            total_values.extend(list(h_v.values()))

        sorted_value = sorted(total_values, reverse=self.pick_high)
        thres_idx = int(math.floor(self.percentiles * len(sorted_value)))
        self.value_threshold = sorted_value[thres_idx]


class UniqueValueFilter(FilterMethod):
    """
    filter the columns if all values in this feature is the same

    Parameters
    ----------
    param : UniqueValueParam object,
            Parameters that user set.

    cols : list of string or -1
            Specify which column(s) need to apply binning. -1 means do binning for all columns.

    statics_obj : MultivariateStatisticalSummary object, default: None
            If those static information has been compute. This can be use as parameter so that no need to
            compute again.

    """

    def __init__(self, param: UniqueValueParam,
                 cols,
                 statics_obj=None):
        super(UniqueValueFilter, self).__init__()
        self.eps = param.eps
        self.cols = cols
        self.statics_obj = statics_obj

    def fit(self, data_instances):
        self._init_cols(data_instances)

        if self.statics_obj is None:
            self.statics_obj = MultivariateStatisticalSummary(data_instances, self.cols)

        left_cols = {}
        max_values = self.statics_obj.get_max()
        min_values = self.statics_obj.get_min()

        for col_name in self.cols:
            min_max_diff = math.fabs(max_values[col_name] - min_values[col_name])
            if min_max_diff >= self.eps:
                left_cols[col_name] = True
            else:
                left_cols[col_name] = False
            self.feature_values[col_name] = min_max_diff

        self.left_cols = left_cols
        self.left_cols = self._keep_one_feature(pick_high=False)
        return left_cols

    def get_param_obj(self):
        left_col_obj = feature_selection_param_pb2.LeftCols(original_cols=self.cols,
                                                            left_cols=self.left_cols)

        result = feature_selection_param_pb2.FeatureSelectionFilterParam(feature_values=self.feature_values,
                                                                         left_cols=left_col_obj,
                                                                         filter_name=consts.UNIQUE_VALUE)
        return result

    def get_meta_obj(self):
        result = feature_selection_meta_pb2.UniqueValueMeta(eps=self.eps)
        return result


class IVValueSelectFilter(FilterMethod):
    """
    Drop the columns if their iv is smaller than a threshold

    Parameters
    ----------
    param : IVSelectionParam object,
            Parameters that user set.

    cols : list of string
            Specify header of guest variances.

    binning_obj : Binning object
        Use for collecting iv among all parties.
    """

    def __init__(self, param, cols, binning_obj):
        super(IVValueSelectFilter, self).__init__()
        self.value_threshold = param.value_threshold
        self.cols = cols
        self.binning_obj = binning_obj

    def fit(self, data_instances=None):
        # fit guest
        guest_binning_result = self.binning_obj.binning_result
        for col_name, iv_attr in guest_binning_result.items():
            self.feature_values[col_name] = iv_attr.iv

        self.left_cols = self.filter_one_party(self.feature_values, True, self.value_threshold)
        self.left_cols = self._keep_one_feature()

        for host_name, host_bin_result in self.binning_obj.host_results.items():
            tmp_host_value = {}
            for host_col_name, host_iv_attr in host_bin_result.items():
                tmp_host_value[host_col_name] = host_iv_attr.iv
            self.host_feature_values[host_name] = tmp_host_value
            left_cols = self.filter_one_party(tmp_host_value, True, self.value_threshold)
            left_cols = self._keep_one_feature(left_cols=left_cols, feature_values=tmp_host_value)
            self.host_cols[host_name] = left_cols

        return self.left_cols

    def get_param_obj(self):
        left_col_obj = feature_selection_param_pb2.LeftCols(original_cols=self.cols,
                                                            left_cols=self.left_cols)
        host_obj = {}
        for host_name, host_left_cols in self.host_cols.items():
            host_cols = list(host_left_cols.keys())
            left_col_obj = feature_selection_param_pb2.LeftCols(original_cols=host_cols,
                                                                left_cols=host_left_cols)
            host_obj[host_name] = left_col_obj

        host_value_objs = {}
        for host_name, host_feature_values in self.host_feature_values.items():
            host_feature_value_obj = feature_selection_param_pb2.FeatureValue(feature_values=host_feature_values)
            host_value_objs[host_name] = host_feature_value_obj

        result = feature_selection_param_pb2.FeatureSelectionFilterParam(feature_values=self.feature_values,
                                                                         host_feature_values=host_value_objs,
                                                                         left_cols=left_col_obj,
                                                                         host_left_cols=host_obj,
                                                                         filter_name=consts.IV_VALUE_THRES)
        return result

    def get_meta_obj(self):
        result = feature_selection_meta_pb2.IVValueSelectionMeta(value_threshold=self.value_threshold)
        return result


class IVPercentileFilter(FilterMethod):
    """
    Drop the columns if their iv is smaller than a threshold of percentile.

    Parameters
    ----------
    iv_param : IVSelectionParam object,
            Parameters that user set.

    cols : list of string
            Specify header of guest variances.

    binning_obj : Binning object
        Use for collecting iv among all parties.
    """

    def __init__(self, iv_param, cols, host_cols, binning_obj):
        super(IVPercentileFilter, self).__init__()
        self.iv_param = iv_param
        self.percentile_thres = iv_param.percentile_threshold
        self.cols = cols
        self.host_cols = host_cols
        self.binning_obj = binning_obj

    def fit(self, data_instances=None):

        # fit guest
        guest_binning_result = self.binning_obj.binning_result
        for col_name, iv_attr in guest_binning_result.items():
            if col_name not in self.cols:
                continue
            self.feature_values[col_name] = iv_attr.iv

        host_feature_values = {}
        for host_name, host_bin_result in self.binning_obj.host_results.items():
            if host_name not in self.host_cols:
                continue
            else:
                host_to_select_cols = self.host_cols.get(host_name)
            tmp_host_value = {}
            for host_col_name, host_iv_attr in host_bin_result.items():
                if host_col_name not in host_to_select_cols:
                    continue
                tmp_host_value[host_col_name] = host_iv_attr.iv
            host_feature_values[host_name] = tmp_host_value
            self.host_feature_values = host_feature_values

        union_filter = UnionPercentileFilter(local_variance=self.feature_values,
                                             host_variances=host_feature_values,
                                             percentile=self.percentile_thres,
                                             pick_high=True)
        local_left_cols, host_left_cols = union_filter.fit(data_instances)
        self.left_cols = local_left_cols
        self.host_cols = host_left_cols
        return self.left_cols

    def get_param_obj(self):
        left_col_obj = feature_selection_param_pb2.LeftCols(original_cols=self.cols,
                                                            left_cols=self.left_cols)
        host_obj = {}
        for host_name, host_left_cols in self.host_cols.items():
            host_cols = list(host_left_cols.keys())
            left_col_obj = feature_selection_param_pb2.LeftCols(original_cols=host_cols,
                                                                left_cols=host_left_cols)
            host_obj[host_name] = left_col_obj

        host_value_objs = {}
        for host_name, host_feature_values in self.host_feature_values.items():
            host_feature_value_obj = feature_selection_param_pb2.FeatureValue(feature_values=host_feature_values)
            host_value_objs[host_name] = host_feature_value_obj

        result = feature_selection_param_pb2.FeatureSelectionFilterParam(feature_values=self.feature_values,
                                                                         host_feature_values=host_value_objs,
                                                                         left_cols=left_col_obj,
                                                                         host_left_cols=host_obj,
                                                                         filter_name=consts.IV_PERCENTILE)
        return result

    def get_meta_obj(self):
        result = feature_selection_meta_pb2.IVPercentileSelectionMeta(percentile_threshold=self.percentile_thres)
        return result


class CoeffOfVarValueFilter(FilterMethod):
    """
    Drop the columns if their coefficient of varaiance is smaller than a threshold.

    Parameters
    ----------
    param : CoeffOfVarSelectionParam object,
            Parameters that user set.

    cols : list of string
            Specify header of guest variances.

    statics_obj : MultivariateStatisticalSummary object, default: None
            If those static information has been compute. This can be use as parameter so that no need to
            compute again.
    """

    def __init__(self, param, cols, statics_obj=None):
        super(CoeffOfVarValueFilter, self).__init__()
        self.value_threshold = param.value_threshold
        self.cols = cols
        self.statics_obj = statics_obj

    def fit(self, data_instances):
        self._init_cols(data_instances)
        if self.statics_obj is None:
            self.statics_obj = MultivariateStatisticalSummary(data_instances, self.cols)

        std_var = self.statics_obj.get_std_variance(self.cols_dict)
        mean_value = self.statics_obj.get_mean(self.cols_dict)
        for col_name, s_v in std_var.items():
            mean = mean_value[col_name]
            coeff_of_var = math.fabs(s_v / mean)
            self.feature_values[col_name] = coeff_of_var
            if coeff_of_var >= self.value_threshold:
                self.left_cols[col_name] = True
            else:
                self.left_cols[col_name] = False

        self.left_cols = self._keep_one_feature()

        return self.left_cols

    def get_param_obj(self):
        left_col_obj = feature_selection_param_pb2.LeftCols(original_cols=self.cols,
                                                            left_cols=self.left_cols)

        result = feature_selection_param_pb2.FeatureSelectionFilterParam(
            feature_values=self.feature_values,
            left_cols=left_col_obj,
            filter_name=consts.COEFFICIENT_OF_VARIATION_VALUE_THRES)
        return result

    def get_meta_obj(self):
        result = feature_selection_meta_pb2.CoeffOfVarSelectionMeta(value_threshold=self.value_threshold)
        return result


class OutlierFilter(FilterMethod):
    """
    Given percentile and threshold. Judge if this quantile point is larger than threshold. Filter those larger ones.

    Parameters
    ----------
    params : OutlierColsSelectionParam object,
            Parameters that user set.

    cols : int or list of int
            Specify which column(s) need to apply this filter method. -1 means do binning for all columns.

    """

    def __init__(self, params, cols):
        super(OutlierFilter, self).__init__()
        self.percentile = params.percentile
        self.upper_threshold = params.upper_threshold
        self.cols = cols

    def fit(self, data_instances, bin_param=None):
        if bin_param is None:  # Use default setting
            bin_param = FeatureBinningParam()

        bin_obj = QuantileBinning(bin_param)
        query_result = bin_obj.query_quantile_point(data_instances, self.cols, self.percentile)
        for col_name, feature_value in query_result.items():
            self.feature_values[col_name] = feature_value
            if feature_value < self.upper_threshold:
                self.left_cols[col_name] = True
            else:
                self.left_cols[col_name] = False

        self.left_cols = self._keep_one_feature()
        return self.left_cols

    def get_param_obj(self):
        left_col_obj = feature_selection_param_pb2.LeftCols(original_cols=self.cols,
                                                            left_cols=self.left_cols)

        result = feature_selection_param_pb2.FeatureSelectionFilterParam(feature_values=self.feature_values,
                                                                         left_cols=left_col_obj,
                                                                         filter_name=consts.OUTLIER_COLS)
        return result

    def get_meta_obj(self):
        result = feature_selection_meta_pb2.OutlierColsSelectionMeta(percentile=self.percentile,
                                                                     upper_threshold=self.upper_threshold)
        return result

