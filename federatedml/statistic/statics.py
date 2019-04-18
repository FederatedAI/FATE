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

import functools
import math
import sys

from arch.api.utils import log_utils
from federatedml.statistic.data_overview import get_features_shape
from federatedml.feature.binning import QuantileBinning
from federatedml.param.param import FeatureBinningParam

LOGGER = log_utils.getLogger()


class SummaryStatistics(object):
    def __init__(self):
        self.sum = 0
        self.sum_square = 0
        self.max_value = - sys.maxsize - 1
        self.min_value = sys.maxsize
        self.count = 0

    def add_value(self, value):
        self.count += 1
        self.sum += value
        self.sum_square += value ** 2
        if value > self.max_value:
            self.max_value = value
        if value < self.min_value:
            self.min_value = value

    def merge(self, other):
        self.count += other.count
        self.sum += other.sum
        self.sum_square += other.sum_square
        if self.max_value < other.max_value:
            self.max_value = other.max_value

        if self.min_value > other.min_value:
            self.min_value = other.min_value

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def variance(self):
        mean = self.mean
        return self.sum_square / self.count - 2 * self.sum * mean / self.count + mean ** 2

    @property
    def std_variance(self):
        return math.sqrt(self.variance)


class MultivariateStatisticalSummary(object):
    def __init__(self, data_instances, select_cols):
        self.finish_fit = False
        self.summary_statistics = []
        self.median = None
        self.data_instances = data_instances

        if select_cols == -1:
            features_shape = get_features_shape(data_instances)
            if features_shape is None:
                raise RuntimeError('Cannot get feature shape, please check input data')
            select_cols = [i for i in range(features_shape)]

        self.select_cols = select_cols

    def _static_sums(self):
        """
        Statics sum, sum_square, max_value, min_value,
        so that variance is available.
        """
        partition_cal = functools.partial(self.static_in_partition,
                                          select_cols=self.select_cols)
        summary_statistic_list = self.data_instances.mapPartitions(partition_cal)
        self.summary_statistics = summary_statistic_list.reduce(self.aggregate_statics)
        self.finish_fit = True

    @staticmethod
    def static_in_partition(data_instances, select_cols):
        """
        Statics sums, sum_square, max and min value through one traversal

        Parameters
        ----------
        data_instances : DTable
            The input data

        select_cols : list of int
            Specify which column(s) need to apply binning.

        Returns
        -------
        List of SummaryStatistics object

        """
        summary_statistic_list = []
        for _ in select_cols:
            summary_statistic_list.append(SummaryStatistics())

        for k, instances in data_instances:
            features = instances.features
            for idx, col in enumerate(select_cols):
                value = features[col]
                stat_obj = summary_statistic_list[idx]
                stat_obj.add_value(value)

        return summary_statistic_list

    @staticmethod
    def aggregate_statics(s_list1, s_list2):
        if s_list1 is None and s_list2 is None:
            return None
        if s_list1 is None:
            return s_list2
        if s_list2 is None:
            return s_list1
        new_list = []
        for idx, static_1 in enumerate(s_list1):
            static_1.merge(s_list2[idx])
            new_list.append(static_1)
        return new_list

    def get_mean(self, cols=None):
        """
        Return the mean value(s) of the given column

        Parameters
        ----------
        cols : list of int or None, default: None
            Specify which column(s). If None, it will use the select cols defined initialed.

        Returns
        -------
        return a list of result mean. The order is the same as cols.

        """
        return self._prepare_data(cols, "mean")

    def get_median(self, cols=None):
        medians = []

        if cols is None:
            cols = self.select_cols

        if self.median is None:
            self.median = self._get_quantile_median(self.select_cols)

        for col in cols:
            try:
                idx = self.select_cols.index(col)
                medians.append(self.median[idx])
            except ValueError:
                LOGGER.warning("The column {}, has not set in selection parameters."
                               "median values is not available".format(col))
                medians.append(None)

        return medians

    def _get_quantile_median(self, cols):
        bin_param = FeatureBinningParam(bin_num=2)
        binning_obj = QuantileBinning(bin_param)
        split_points = binning_obj.binning(self.data_instances, cols)
        medians = [x[0] for x in split_points]
        return medians

    def get_variance(self, cols=None):
        return self._prepare_data(cols, "variance")

    def get_std_variance(self, cols=None):
        return self._prepare_data(cols, "std_variance")

    def get_max(self, cols=None):
        return self._prepare_data(cols, "max_value")

    def get_min(self, cols=None):
        return self._prepare_data(cols, "min_value")

    def _prepare_data(self, cols, data_type):
        """
        Return the specific static value(s) of the given column

        Parameters
        ----------
        cols : list of int or None, default: None
            Specify which column(s). If None, it will use the select cols defined initialed.

        data_type : str, "mean", "variance", "std_variance", "max_value" or "mim_value"
            Specify which type to show.

        Returns
        -------
        return a list of result result. The order is the same as cols.
        """
        if not self.finish_fit:
            self._static_sums()

        if cols is None:
            cols = self.select_cols

        result = []
        for col in cols:
            try:
                idx = self.select_cols.index(col)
                summary_obj = self.summary_statistics[idx]
                if data_type == 'mean':
                    result.append(summary_obj.mean)
                elif data_type == 'variance':
                    result.append(summary_obj.variance)
                elif data_type == 'max_value':
                    result.append(summary_obj.max_value)
                elif data_type == 'min_value':
                    result.append(summary_obj.min_value)
                elif data_type == 'std_variance':
                    result.append(summary_obj.std_variance)

            except ValueError:
                LOGGER.warning("The column {}, has not set in selection parameters."
                               "{} values is not available".format(col, data_type))
                result.append(None)
        return result
