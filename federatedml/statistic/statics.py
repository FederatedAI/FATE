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
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.feature.binning.quantile_summaries import QuantileSummaries
from federatedml.feature.instance import Instance
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.feature.sparse_vector import SparseVector
from federatedml.statistic import data_overview
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class SummaryStatistics(object):
    def __init__(self, abnormal_list=None):
        if abnormal_list is None:
            self.abnormal_list = []
        else:
            self.abnormal_list = abnormal_list

        self.sum = 0
        self.sum_square = 0
        self.max_value = - sys.maxsize - 1
        self.min_value = sys.maxsize
        self.count = 0

    def add_value(self, value):
        if value in self.abnormal_list:
            return

        try:
            value = float(value)
        except TypeError:
            LOGGER.warning('The value {} cannot be converted to float'.format(value))
            return

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
        variance = self.sum_square / self.count - mean ** 2
        if math.fabs(variance) < consts.FLOAT_ZERO:
            return 0.0
        return variance

    @property
    def std_variance(self):
        if math.fabs(self.variance) < consts.FLOAT_ZERO:
            return 0.0
        return math.sqrt(self.variance)


class MultivariateStatisticalSummary(object):
    """

    """

    def __init__(self, data_instances, cols_index=-1, abnormal_list=None):
        self.finish_fit_statics = False     # Use for static data
        self.finish_fit_summaries = False   # Use for quantile data
        self.summary_statistics = {}
        self.quantile_summary_dict = {}
        self.cols_dict = {}
        self.medians = None
        self.data_instances = data_instances
        self.cols_index = cols_index
        if abnormal_list is None:
            self.abnormal_list = []
        else:
            self.abnormal_list = abnormal_list
        self._init_cols(data_instances)

        self.label_summary = None

    def _init_cols(self, data_instances):

        # Already initialized
        if len(self.cols_dict) != 0:
            return

        header = data_overview.get_header(data_instances)
        self.header = header
        if self.cols_index == -1:
            self.cols = header
            self.cols_index = [i for i in range(len(header))]
        else:
            cols = []
            for idx in self.cols_index:
                try:
                    idx = int(idx)
                except ValueError:
                    raise ValueError("In binning module, selected index: {} is not integer".format(idx))

                if idx >= len(header):
                    raise ValueError(
                        "In binning module, selected index: {} exceed length of data dimension".format(idx))
                cols.append(header[idx])
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
                                          cols_dict=self.cols_dict,
                                          abnormal_list=self.abnormal_list)
        summary_statistic_dict = self.data_instances.mapPartitions(partition_cal)
        self.summary_statistics = summary_statistic_dict.reduce(self.aggregate_statics)
        self.finish_fit_statics = True

    def _static_quantile_summaries(self):
        """
        Static summaries so that can query a specific quantile point
        """
        if self.finish_fit_summaries is True:
            return

        partition_cal = functools.partial(self.static_summaries_in_partition,
                                          cols_dict=self.cols_dict,
                                          abnormal_list=self.abnormal_list)
        quantile_summary_dict = self.data_instances.mapPartitions(partition_cal)
        self.quantile_summary_dict = quantile_summary_dict.reduce(self.aggregate_statics)
        self.finish_fit_summaries = True

    @staticmethod
    def static_in_partition(data_instances, cols_dict, abnormal_list):
        """
        Statics sums, sum_square, max and min value through one traversal

        Parameters
        ----------
        data_instances : DTable
            The input data

        cols_dict : dict
            Specify which column(s) need to apply statistic.

        abnormal_list: list
            Specify which values are not permitted.

        Returns
        -------
        Dict of SummaryStatistics object

        """
        summary_statistic_dict = {}
        for col_name in cols_dict:
            summary_statistic_dict[col_name] = SummaryStatistics(abnormal_list)

        for k, instances in data_instances:
            if isinstance(instances, Instance):
                features = instances.features
            else:
                features = instances

            if isinstance(features, SparseVector):
                is_sparse = True
            else:
                is_sparse = False

            for col_name, col_index in cols_dict.items():
                if is_sparse:
                    sparse_data = features.get_sparse_vector()
                    value = sparse_data.get(col_index, 0)
                else:
                    value = features[col_index]
                stat_obj = summary_statistic_dict[col_name]
                stat_obj.add_value(value)

        return summary_statistic_dict

    @staticmethod
    def static_summaries_in_partition(data_instances, cols_dict, abnormal_list):
        """
        Statics sums, sum_square, max and min value through one traversal

        Parameters
        ----------
        data_instances : DTable
            The input data

        cols_dict : dict
            Specify which column(s) need to apply statistic.

        abnormal_list: list
            Specify which values are not permitted.

        Returns
        -------
        Dict of SummaryStatistics object

        """
        summary_dict = {}
        for col_name in cols_dict:
            summary_dict[col_name] = QuantileSummaries(abnormal_list=abnormal_list)

        for k, instances in data_instances:
            if isinstance(instances, Instance):
                features = instances.features
            else:
                features = instances

            for col_name, col_index in cols_dict.items():
                value = features[col_index]
                summary_obj = summary_dict[col_name]
                summary_obj.insert(value)

        return summary_dict

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

    def get_quantile_point(self, quantile, cols_dict=None):
        """
        Return the specific quantile point value

        Parameters
        ----------
        quantile : float, 0 <= quantile <= 1
            Specify which column(s) need to apply statistic.

        cols_dict : dict
            Specify which column(s) need to apply statistic.

        Returns
        -------
        return a dict of result quantile points.
        eg.
        quantile_point = {"x1": 3, "x2": 5... }
        """
        quantile_points = {}

        if cols_dict is None:
            cols_dict = self.cols_dict

        self._static_quantile_summaries()

        for col_name in cols_dict:
            if col_name not in self.quantile_summary_dict:
                LOGGER.warning("The column {}, has not set in selection parameters."
                               "Quantile point query is not available".format(col_name))
                continue
            summary_obj = self.quantile_summary_dict[col_name]
            quantile_point = summary_obj.query(quantile)
            quantile_points[col_name] = quantile_point
        return quantile_points

    def _get_quantile_median(self):
        cols_index = self._get_cols_index()
        bin_param = FeatureBinningParam(bin_num=2, bin_indexes=cols_index)
        binning_obj = QuantileBinning(bin_param, abnormal_list=self.abnormal_list)
        split_points = binning_obj.fit_split_points(self.data_instances)
        medians = {}
        for col_name, split_point in split_points.items():
            medians[col_name] = split_point[0]
        return medians

    def _get_cols_index(self):
        cols_index = []
        for col in self.cols:
            idx = self.cols_dict[col]
            cols_index.append(idx)
        return cols_index

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
        if not self.finish_fit_statics:
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

    @staticmethod
    def get_label_static_dict(data_instances):
        result_dict = {}
        for instance in data_instances:
            label_key = instance[1].label
            if label_key not in result_dict:
                result_dict[label_key] = 1
            else:
                result_dict[label_key] += 1
        return result_dict

    @staticmethod
    def merge_result_dict(dict_a, dict_b):
        for k, v in dict_b.items():
            if k in dict_a:
                dict_a[k] += v
            else:
                dict_a[k] = v
        return dict_a

    def get_label_histogram(self):
        label_histogram = self.data_instances.mapPartitions(self.get_label_static_dict).reduce(self.merge_result_dict)
        return label_histogram



