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
import random

from arch.api.proto.feature_selection_param_pb2 import FeatureSelectionFilterParam, FeatureSelectionParam
from arch.api.utils import log_utils
from federatedml.feature.binning import QuantileBinning
from federatedml.param.param import FeatureSelectionParam, IVSelectionParam, FeatureBinningParam, UniqueValueParam
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.util import consts
from federatedml.statistic.data_overview import get_features_shape

LOGGER = log_utils.getLogger()


class FilterMethod(object):
    def filter(self, data_instances):
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

    def _keep_one_feature(self, original_cols, left_cols):
        """
        Make sure at least one feature can be left after filtering.

        Parameters
        ----------
        original_cols : list,
            Column index that before filtering

        left_cols : list,
            Column index that after filtering.

        Returns
        -------
        A list of index of columns left.
        """
        if len(left_cols) >= 1:
            return left_cols
        left_col = random.choice(original_cols)
        return [left_col]


class UniqueValueFilter(FilterMethod):
    """
    filter the columns if all values in this feature is the same

    Parameters
    ----------
    param : UniqueValueParam object,
            Parameters that user set.

    cols : int or list of int
            Specify which column(s) need to apply binning. -1 means do binning for all columns.
    """

    def __init__(self, param: UniqueValueParam,
                 select_cols,
                 statics_obj=None):
        self.eps = param.eps
        self.select_cols = select_cols
        self.statics_obj = statics_obj
        self.left_cols = None

    def filter(self, data_instances):
        if self.statics_obj is None:
            self.statics_obj = MultivariateStatisticalSummary(data_instances, self.select_cols)

        left_cols = []
        max_values = self.statics_obj.get_max(self.select_cols)
        min_values = self.statics_obj.get_min(self.select_cols)

        for idx, col in enumerate(self.select_cols):
            if math.fabs(max_values[idx] - min_values[idx]) >= self.eps:
                left_cols.append(col)

        left_cols = self._keep_one_feature(self.select_cols, left_cols)
        self.left_cols = left_cols
        return left_cols

    def to_result(self):
        params = {'eps': self.eps}
        result = FeatureSelectionFilterParam(param_set=params,
                                             original_cols=self.select_cols,
                                             left_cols=self.left_cols,
                                             filter_name=consts.UNIQUE_VALUE)
        return result


class IVValueSelectFilter(FilterMethod):
    """
    Drop the columns if their iv is smaller than a threshold

    Parameters
    ----------
    param : IVSelectionParam object,
            Parameters that user set.

    cols : int or list of int
            Specify which column(s) need to apply binning. -1 means do binning for all columns.

    iv_attrs : list of IVAttributes object, default: None
            If provided, the filter method will omit binning method and used the given IVAttributes instead.

    """
    def __init__(self, param, select_cols, iv_attrs=None):
        self.value_threshold = param.value_threshold
        self.select_cols = select_cols
        self.iv_attrs = iv_attrs
        self.bin_param = param.bin_param
        self.bin_param.cols = self.select_cols
        self.left_cols = None

    def filter(self, data_instances=None):
        if self.iv_attrs is None and data_instances is None:
            raise RuntimeError("In iv value filter, iv_attrs and data_instances cannot be None simultaneously")

        if self.iv_attrs is None:
            binning_obj = QuantileBinning(self.bin_param)
            self.iv_attrs = binning_obj.cal_local_iv(data_instances, cols=self.select_cols)

        ivs = [x.iv for x in self.iv_attrs]
        left_cols = []
        for idx, col in enumerate(self.select_cols):
            iv = ivs[idx]
            if iv >= self.value_threshold:
                left_cols.append(col)

        left_cols = self._keep_one_feature(self.select_cols, left_cols)
        self.left_cols = left_cols
        return left_cols

    def to_result(self):
        params = {'value_threshold': self.value_threshold}
        result = FeatureSelectionFilterParam(param_set=params,
                                             original_cols=self.select_cols,
                                             left_cols=self.left_cols,
                                             filter_name=consts.IV_VALUE_THRES)
        return result


class IVPercentileFilter(FilterMethod):
    """
    Drop the columns if their iv is smaller than a threshold of percentile.

    Parameters
    ----------
    param : IVSelectionParam object,
            Parameters that user set.

    cols : int or list of int
            Specify which column(s) need to apply binning. -1 means do binning for all columns.
    """
    def __init__(self, iv_param, cols=None):
        self.iv_param = iv_param
        self.percentile_thres = iv_param.percentile_threshold
        self.all_iv_attrs = []
        self.party_cols = []
        if cols is not None:
            self.party_cols.append(cols)
        self.bin_param = iv_param.bin_param

        self.left_cols = None

    def add_attrs(self, iv_attrs, cols=None):
        self.all_iv_attrs.append(iv_attrs)
        if cols is None:
            cols = [i for i in range(len(iv_attrs))]
        self.party_cols.append(cols)

    # Use when cols has set but iv haven't added.
    def add_attrs_only(self, iv_attrs):
        self.all_iv_attrs.append(iv_attrs)

    def filter_multiple_parties(self, data_instances=None):
        """
        Used in federated iv filtering. This function will put iv of both guest and host party together, and judge
        which columns satisfy the parameters setting. For example, if percentile has been set as 0.8, and there are
        20 features in guest and 20 in host. This filter gonna sort these 40 features' iv and pick
        floor((20 + 20) * 0.8) = 32nd highest iv as value threshold and then put into IV value filter.

        Therefore, the number of left columns maybe larger than floor(total_iv * percentile) if multiple columns of
        iv are same.

        Besides, if only iv attrs of one party has been provided. This function also works as normal filter.

        Parameters
        ----------
        data_instances : DTable,
            Input data

        Returns
        -------
        left_cols : List,
            Contains the left cols of both parties. E.g. [[0,2,3,4], [2,3,5,6]]. The first element in list represent
            for left columns of guest and the second one represent for host.
        """
        if len(self.all_iv_attrs) == 0 and data_instances is None:
            raise RuntimeError("In iv value filter, iv_attrs and data_instances cannot be None simultaneously")

        # If already set cols but without iv_attrs, use binning object get iv_attrs first
        if len(self.all_iv_attrs) == 0 and len(self.party_cols) == 1:
            binning_obj = QuantileBinning(self.bin_param)
            self.all_iv_attrs.append(binning_obj.cal_local_iv(data_instances, cols=self.party_cols[0]))

        thres_iv = self._get_real_iv_thres()
        new_iv_param = IVSelectionParam(value_threshold=thres_iv)
        left_cols = []

        for idx, iv_attrs in enumerate(self.all_iv_attrs):
            cols = self.party_cols[idx]
            tmp_iv_thres_obj = IVValueSelectFilter(new_iv_param, select_cols=cols, iv_attrs=iv_attrs)
            party_left_cols = tmp_iv_thres_obj.filter()
            left_cols.append(party_left_cols)
            LOGGER.debug("left_cols: {}".format(left_cols))

        self.left_cols = self._keep_one_feature(self.party_cols[0], left_cols[0])

        # self.left_cols = left_cols  # Record guest party only
        return left_cols

    def _get_real_iv_thres(self):
        all_ivs = []
        for iv_attrs in self.all_iv_attrs:
            all_ivs.extend([x.iv for x in iv_attrs])
        all_ivs = sorted(all_ivs, reverse=True)
        thres_idx = int(math.floor(self.percentile_thres * len(all_ivs)))
        if thres_idx == len(all_ivs):
            thres_idx -= 1

        thres_iv = all_ivs[thres_idx]
        return thres_iv

    def to_result(self):
        params = {'percentile_thres': self.percentile_thres}
        result = FeatureSelectionFilterParam(param_set=params,
                                             original_cols=self.party_cols[0],
                                             left_cols=self.left_cols,
                                             filter_name=consts.IV_PERCENTILE)
        return result


class CoeffOfVarValueFilter(FilterMethod):
    """
    Drop the columns if their coefficient of varaiance is smaller than a threshold.

    Parameters
    ----------
    param : CoeffOfVarSelectionParam object,
            Parameters that user set.

    select_cols : int or list of int
            Specify which column(s) need to apply this filter method. -1 means do binning for all columns.

    statics_obj : MultivariateStatisticalSummary object, default: None
            If those static information has been compute. This can be use as parameter so that no need to
            compute again.
    """
    def __init__(self, param, select_cols, statics_obj=None):
        self.value_threshold = param.value_threshold
        self.select_cols = select_cols
        self.statics_obj = statics_obj
        self.left_cols = None

    def filter(self, data_instances):
        if self.statics_obj is None:
            self.statics_obj = MultivariateStatisticalSummary(data_instances, self.select_cols)

        left_cols = []
        std_var = self.statics_obj.get_std_variance(self.select_cols)
        mean_value = self.statics_obj.get_mean(self.select_cols)
        for idx, s_v in enumerate(std_var):
            mean = mean_value[idx]
            coeff_of_var = math.fabs(s_v / mean)

            if coeff_of_var >= self.value_threshold:
                left_cols.append(self.select_cols[idx])

        left_cols = self._keep_one_feature(self.select_cols, left_cols)
        self.left_cols = left_cols
        return left_cols

    def to_result(self):
        params = {'value_threshold': self.value_threshold}
        result = FeatureSelectionFilterParam(param_set=params,
                                             original_cols=self.select_cols,
                                             left_cols=self.left_cols,
                                             filter_name=consts.COEFFICIENT_OF_VARIATION_VALUE_THRES)
        return result


class OutlierFilter(FilterMethod):
    """
    Given percentile and threshold. Judge if this quantile point is larger than threshold. Filter those larger ones.

    Parameters
    ----------
    param : OutlierColsSelectionParam object,
            Parameters that user set.

    select_cols : int or list of int
            Specify which column(s) need to apply this filter method. -1 means do binning for all columns.

    statics_obj : MultivariateStatisticalSummary object, default: None
            If those static information has been compute. This can be use as parameter so that no need to
            compute again.
    """
    def __init__(self, params, select_cols):
        self.percentile = params.percentile
        self.upper_threshold = params.upper_threshold
        self.select_cols = select_cols
        self.left_cols = None

    def filter(self, data_instances, bin_param=None):
        if bin_param is None:  # Use default setting
            bin_param = FeatureBinningParam()

        bin_obj = QuantileBinning(bin_param)
        query_result = bin_obj.query_quantile_point(data_instances, self.select_cols, self.percentile)
        left_cols = []
        for idx, q_r in enumerate(query_result):
            if q_r < self.upper_threshold:
                left_cols.append(self.select_cols[idx])

        left_cols = self._keep_one_feature(self.select_cols, left_cols)
        self.left_cols = left_cols
        return left_cols

    def to_result(self):
        params = {'percentile': self.percentile,
                  'upper_threshold': self.upper_threshold}
        result = FeatureSelectionFilterParam(param_set=params,
                                             original_cols=self.select_cols,
                                             left_cols=self.left_cols,
                                             filter_name=consts.OUTLIER_COLS)
        return result


class FeatureSelection(object):
    """
    Made feature selection based on set parameters.

    Parameters
    ----------
    params: float, FeatureSelectionParam
        User defined parameters.

    """

    def __init__(self, params: FeatureSelectionParam):
        self.filter_methods = params.filter_method
        self.select_cols = params.select_cols
        self.left_cols = None
        self.params = params
        self.static_obj = None
        self.iv_attrs = None
        self.results = []

    def filter(self, data_instances):
        if self.select_cols == -1:
            features_shape = get_features_shape(data_instances)
            if features_shape is None:
                raise RuntimeError('Cannot get feature shape, please check input data')
            self.select_cols = [i for i in range(features_shape)]
        self.left_cols = self.select_cols.copy()

        for method in self.filter_methods:
            self.filter_one_method(data_instances, method)
        return self.left_cols

    def filter_one_method(self, data_instances, method):
        """
        Given data and method, perform the filter methods.

        Parameters
        ----------
        data_instances: DTable,
            Input data

        method : str
            The method name. if the name is not in

        """

        if method == consts.UNIQUE_VALUE:
            filter_obj = UniqueValueFilter(self.params.unique_param,
                                           select_cols=self.left_cols,
                                           statics_obj=self.static_obj)
            self.left_cols = filter_obj.filter(data_instances)
            self.static_obj = filter_obj.statics_obj
            self.results.append(filter_obj.to_result())

        elif method == consts.IV_VALUE_THRES:
            filter_obj = IVValueSelectFilter(self.params.iv_param,
                                             select_cols=self.left_cols,
                                             iv_attrs=self.iv_attrs)
            self.left_cols = filter_obj.filter(data_instances)
            self.iv_attrs = filter_obj.iv_attrs
            self.results.append(filter_obj.to_result())

        elif method == consts.IV_PERCENTILE:
            filter_obj = IVPercentileFilter(self.params.iv_param,
                                            cols=self.left_cols)
            if self.iv_attrs is not None:
                filter_obj.add_attrs_only(self.iv_attrs)
            self.left_cols = filter_obj.filter_multiple_parties(data_instances)[0]
            self.iv_attrs = filter_obj.all_iv_attrs[0]
            self.results.append(filter_obj.to_result())

        elif method == consts.COEFFICIENT_OF_VARIATION_VALUE_THRES:
            filter_obj = CoeffOfVarValueFilter(self.params.coe_param,
                                               select_cols=self.left_cols,
                                               statics_obj=self.static_obj)
            self.left_cols = filter_obj.filter(data_instances)
            self.static_obj = filter_obj.statics_obj
            self.results.append(filter_obj.to_result())

        elif method == consts.OUTLIER_COLS:
            filter_obj = OutlierFilter(self.params.outlier_param,
                                       select_cols=self.left_cols)
            self.left_cols = filter_obj.filter(data_instances)
            self.results.append(filter_obj.to_result())

    def set_iv_attrs(self, iv_attrs):
        self.iv_attrs = iv_attrs

    def set_static_obj(self, static_obj):
        self.static_obj = static_obj

    def to_result(self):
        result = FeatureSelectionParam(results=self.results)
        return result
