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
from federatedml.statistic.data_overview import get_header
from federatedml.feature.binning.quantile_binning import QuantileBinning
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
    def __init__(self, data_instances, cols):
        self.finish_fit = False
        self.summary_statistics = []
        self.medians = None
        self.data_instances = data_instances

        header = get_header(data_instances)

        if cols == -1:
            self.cols = header
        else:
            self.cols = cols

        self.cols_dict = {}
        for col in self.cols:
            col_index = header.index(col)
            self.cols_dict[col] = col_index

    def _static_sums(self):
        """
        Statics sum, sum_square, max_value, min_value,
        so that variance is available.
        """
        partition_cal = functools.partial(self.static_in_partition,
                                          cols_dict=self.cols_dict)
        summary_statistic_list = self.data_instances.mapPartitions(partition_cal)
        self.summary_statistics = summary_statistic_list.reduce(self.aggregate_statics)
        self.finish_fit = True

    @staticmethod
    def static_in_partition(data_instances, cols_dict):
        """
        Statics sums, sum_square, max and min value through one traversal

        Parameters
        ----------
        data_instances : DTable
            The input data

        cols_dict : dict
            Specify which column(s) need to apply statistic.

        Returns
        -------
        Dict of SummaryStatistics object

        """
        summary_statistic_dict = {}
        for col_name in cols_dict:
            summary_statistic_dict[col_name] = SummaryStatistics()

        for k, instances in data_instances:
            features = instances.features
            for col_name, col_index in cols_dict.items():
                value = features[col_index]
                stat_obj = summary_statistic_dict[col_name]
                stat_obj.add_value(value)

        return summary_statistic_dict

    @staticmethod
    def aggregate_statics(s_dict1, s_dict2):
        if s_dict1 is None and s_dict2 is None:
            return None
        if s_dict1 is None:
            return s_dict2
        if s_dict2 is None:
            return s_dict1

        new_dict = {}
        for col_name, static_1 in s_dict1.items():
            static_1.merge(s_dict2[col_name])
            new_dict[col_name] = static_1
        return new_dict

    def get_mean(self, cols_dict=None):
        """
        Return the mean value(s) of the given column

        Parameters
        ----------
        cols_dict : dict
            Specify which column(s) need to apply statistic.

        Returns
        -------
        return a dict of result mean.

        """
        return self._prepare_data(cols_dict, "mean")

    def get_median(self, cols_dict=None):
        medians = {}

        if cols_dict is None:
            cols_dict = self.cols_dict

        if self.medians is None:
            self.medians = self._get_quantile_median()

        for col_name in cols_dict:
            if col_name not in self.medians:
                LOGGER.warning("The column {}, has not set in selection parameters."
                               "median values is not available".format(col_name))
                continue
            medians[col_name] = self.medians[col_name]

        return medians

    def _get_quantile_median(self):
        bin_param = FeatureBinningParam(bin_num=2, cols=self.cols)
        binning_obj = QuantileBinning(bin_param)
        split_points = binning_obj.fit_split_points(self.data_instances)
        medians = {}
        for col_name, split_point in split_points.items():
            medians[col_name] = split_point[0]
        return medians

    def get_variance(self, cols_dict=None):
        return self._prepare_data(cols_dict, "variance")

    def get_std_variance(self, cols_dict=None):
        return self._prepare_data(cols_dict, "std_variance")

    def get_max(self, cols_dict=None):
        return self._prepare_data(cols_dict, "max_value")

    def get_min(self, cols_dict=None):
        return self._prepare_data(cols_dict, "min_value")

    def _prepare_data(self, cols_dict, data_type):
        """
        Return the specific static value(s) of the given column

        Parameters
        ----------
        cols_dict : dict
            Specify which column(s) need to apply statistic.

        data_type : str, "mean", "variance", "std_variance", "max_value" or "mim_value"
            Specify which type to show.

        Returns
        -------
        return a list of result result. The order is the same as cols.
        """
        if not self.finish_fit:
            self._static_sums()

        if cols_dict is None:
            cols_dict = self.cols_dict

        result = {}
        for col_name, col_index in cols_dict.items():
            if col_name not in self.cols_dict:
                LOGGER.warning("feature {} has not been static yet. Has been skipped".format(col_name))
                continue

            summary_obj = self.summary_statistics[col_name]
            if data_type == 'mean':
                result[col_name] = summary_obj.mean
            elif data_type == 'variance':
                result[col_name] = summary_obj.variance
            elif data_type == 'max_value':
                result[col_name] = summary_obj.max_value
            elif data_type == 'min_value':
                result[col_name] = summary_obj.min_value
            elif data_type == 'std_variance':
                result[col_name] = summary_obj.std_variance

        return result
