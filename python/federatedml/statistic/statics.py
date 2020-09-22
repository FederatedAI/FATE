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

import copy
import functools
import math

import numpy as np

from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.feature.binning.quantile_summaries import QuantileSummaries
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.statistic import data_overview
# from federatedml.statistic.feature_statistic import feature_statistic
from federatedml.util import LOGGER
from federatedml.util import consts


class SummaryStatistics(object):
    def __init__(self, length, abnormal_list=None, stat_order=2, bias=True):
        self.abnormal_list = abnormal_list
        self.sum = np.zeros(length)
        self.sum_square = np.zeros(length)
        self.max_value = -np.inf * np.ones(length)
        self.min_value = np.inf * np.ones(length)
        self.count = np.zeros(length)
        self.length = length
        self.stat_order = stat_order
        self.bias = bias
        m = 3
        while m <= stat_order:
            exp_sum_m = np.zeros(length)
            setattr(self, f"exp_sum_{m}", exp_sum_m)
            m += 1

    def add_rows(self, rows):
        """
        When getting E(x^n), the formula are:
        .. math::

            (i-1)/i * S_{i-1} + 1/i * x_i

        where i is the current count, and S_i is the current expectation of x
        """
        rows = np.array(rows, dtype=float)
        if self.abnormal_list is None:
            self.count += 1
            self.sum += rows
            self.sum_square += rows ** 2
            self.max_value = np.max([self.max_value, rows], axis=0)
            self.min_value = np.min([self.min_value, rows], axis=0)
            for m in range(3, self.stat_order + 1):
                exp_sum_m = getattr(self, f"exp_sum_{m}")
                exp_sum_m = (self.count - 1) / self.count * exp_sum_m + rows ** m / self.count
                setattr(self, f"exp_sum_{m}", exp_sum_m)
        else:
            for idx, value in enumerate(rows):
                if value in self.abnormal_list:
                    continue
                self.count[idx] += 1
                self.sum[idx] += value
                self.sum_square[idx] += value ** 2
                self.max_value[idx] = np.max([self.max_value[idx], value])
                self.min_value[idx] = np.min([self.min_value[idx], value])
                for m in range(3, self.stat_order + 1):
                    exp_sum_m = getattr(self, f"exp_sum_{m}")
                    exp_sum_m[idx] = (self.count[idx] - 1) / self.count[idx] * \
                                     exp_sum_m[idx] + rows[idx] ** m / self.count[idx]
                    setattr(self, f"exp_sum_{m}", exp_sum_m)

    def merge(self, other):
        if self.stat_order != other.stat_order:
            raise AssertionError("Two merging summary should have same order.")
        self.sum += other.sum
        self.sum_square += other.sum_square
        self.max_value = np.max([self.max_value, other.max_value], axis=0)
        self.min_value = np.min([self.min_value, other.min_value], axis=0)
        for m in range(3, self.stat_order + 1):
            sum_m_1 = getattr(self, f"exp_sum_{m}")
            sum_m_2 = getattr(other, f"exp_sum_{m}")
            exp_sum = (sum_m_1 * self.count + sum_m_2 * other.count) / (self.count + other.count)
            setattr(self, f"exp_sum_{m}", exp_sum)
        self.count += other.count
        return self

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def max(self):
        return self.max_value

    @property
    def min(self):
        return self.min_value

    @property
    def variance(self):
        mean = self.mean
        variance = self.sum_square / self.count - mean ** 2
        variance = np.array([x if math.fabs(x) >= consts.FLOAT_ZERO else 0.0 for x in variance])
        return variance

    @property
    def coefficient_of_variance(self):
        mean = np.array([consts.FLOAT_ZERO if math.fabs(x) < consts.FLOAT_ZERO else x \
                         for x in self.mean])
        return np.fabs(self.stddev / mean)

    @property
    def stddev(self):
        return np.sqrt(self.variance)

    @property
    def moment_3(self):
        """
        In mathematics, a moment is a specific quantitative measure of the shape of a function.
        where the k-th central moment of a data sample is:
        .. math::

            m_k = \frac{1}{n} \sum_{i = 1}^n (x_i - \bar{x})^k

        the 3rd central moment is often used to calculate the coefficient of skewness
        """
        if self.stat_order < 3:
            raise ValueError("The third order of expectation sum has not been statistic.")
        exp_sum_2 = self.sum_square / self.count
        exp_sum_3 = getattr(self, "exp_sum_3")
        mu = self.mean
        return exp_sum_3 - 3 * mu * exp_sum_2 + 2 * mu ** 3

    @property
    def moment_4(self):
        """
        In mathematics, a moment is a specific quantitative measure of the shape of a function.
        where the k-th central moment of a data sample is:
        .. math::

            m_k = \frac{1}{n} \ sum_{i = 1}^n (x_i - \bar{x})^k

        the 4th central moment is often used to calculate the coefficient of kurtosis
        """
        if self.stat_order < 3:
            raise ValueError("The third order of expectation sum has not been statistic.")
        exp_sum_2 = self.sum_square / self.count
        exp_sum_3 = getattr(self, "exp_sum_3")
        exp_sum_4 = getattr(self, "exp_sum_4")
        mu = self.mean
        return exp_sum_4 - 4 * mu * exp_sum_3 + 6 * mu ** 2 * exp_sum_2 - 3 * mu ** 4

    @property
    def skewness(self):
        """
            The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness, i.e.
        .. math::
            g_1=\frac{m_3}{m_2^{3/2}}

        where
        .. math::
            m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i

        If the bias is False, return the adjusted Fisher-Pearson standardized moment coefficient
        i.e.

        .. math::

        G_1=\frac{k_3}{k_2^{3/2}}=
            \frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}.

        """
        m2 = self.variance
        m3 = self.moment_3
        n = self.count

        zero = (m2 == 0)
        np.seterr(divide='ignore', invalid='ignore')
        vals = np.where(zero, 0, m3 / m2 ** 1.5)

        if not self.bias:
            can_correct = (n > 2) & (m2 > 0)
            if can_correct.any():
                m2 = np.extract(can_correct, m2)
                m3 = np.extract(can_correct, m3)
                nval = np.sqrt((n - 1.0) * n) / (n - 2.0) * m3 / m2 ** 1.5
                np.place(vals, can_correct, nval)
        return vals

    @property
    def kurtosis(self):
        """
        Return the sample excess kurtosis which
        .. math::
            g = \frac{m_4}{m_2^2} - 3

        If bias is False, the calculations are corrected for statistical bias.
        """
        m2 = self.variance
        m4 = self.moment_4
        n = self.count
        zero = (m2 == 0)
        np.seterr(divide='ignore', invalid='ignore')
        result = np.where(zero, 0, m4 / m2 ** 2.0)
        if not self.bias:
            can_correct = (n > 3) & (m2 > 0)
            if can_correct.any():
                m2 = np.extract(can_correct, m2)
                m4 = np.extract(can_correct, m4)
                nval = 1.0 / (n - 2) / (n - 3) * ((n ** 2 - 1.0) * m4 / m2 ** 2.0 - 3 * (n - 1) ** 2.0)
                np.place(result, can_correct, nval + 3.0)
        return result - 3


class MissingStatistic(object):

    def __init__(self, missing_val=None):
        super(MissingStatistic, self).__init__()

        self.missing_val = None
        self.feature_summary = {}
        self.missing_feature = []
        self.all_feature_list = []
        self.tag_id_mapping, self.id_tag_mapping = {}, {}
        self.dense_missing_val = missing_val

    @staticmethod
    def is_sparse(tb):
        return type(tb.take(1)[0][1].features) == SparseVector

    @staticmethod
    def check_table_content(tb):

        if not tb.count() > 0:
            raise ValueError('input table must contains at least 1 sample')
        first_ = tb.take(1)[0][1]
        if type(first_) == Instance:
            return True
        else:
            raise ValueError('unknown input format')

    def fit(self, tb):

        LOGGER.debug('start to compute feature lost ratio')

        if not self.check_table_content(tb):
            raise ValueError('contents of input table must be instances of class “Instance"')

        header = tb.schema['header']
        self.all_feature_list = header

        self.tag_id_mapping = {v: k for k, v in enumerate(header)}
        self.id_tag_mapping = {k: v for k, v in enumerate(header)}

        feature_count_rs = self.count_feature_ratio(tb, self.tag_id_mapping, not self.is_sparse(tb),
                                                    missing_val=self.missing_val)
        for idx, count_val in enumerate(feature_count_rs):
            self.feature_summary[self.id_tag_mapping[idx]] = 1 - (count_val / tb.count())
            if (count_val / tb.count()) == 0:
                self.missing_feature.append(self.id_tag_mapping[idx])

        return self.feature_summary

    @staticmethod
    def count_feature_ratio(tb, tag_id_mapping, dense_input, missing_val=None):
        func = functools.partial(MissingStatistic.map_partitions_count, tag_id_mapping=tag_id_mapping,
                                 dense_input=dense_input,
                                 missing_val=missing_val)
        rs = tb.applyPartitions(func)
        return rs.reduce(MissingStatistic.reduce_count_rs)

    @staticmethod
    def map_partitions_count(iterable, tag_id_mapping, dense_input=True, missing_val=None):

        count_arr = np.zeros(len(tag_id_mapping))
        for k, v in iterable:

            # in dense input, missing feature is set as np.nan
            if dense_input:
                feature = v.features  # a numpy array
                arr = np.array(list(feature))
                if missing_val is None:
                    idx_arr = np.argwhere(~np.isnan(arr)).flatten()
                else:
                    idx_arr = np.argwhere(~(arr == missing_val)).flatten()

            # in sparse input, missing features have no key in the dict
            else:
                feature = v.features.sparse_vec  # a dict
                idx_arr = np.array(list(feature.keys()))

            if len(idx_arr) != 0:
                count_arr[idx_arr] += 1

        return count_arr

    @staticmethod
    def reduce_count_rs(arr1, arr2):
        return arr1 + arr2


class MultivariateStatisticalSummary(object):
    """

    """

    def __init__(self, data_instances, cols_index=-1, abnormal_list=None,
                 error=consts.DEFAULT_RELATIVE_ERROR, stat_order=2, bias=True):
        self.finish_fit_statics = False  # Use for static data
        # self.finish_fit_summaries = False   # Use for quantile data
        self.binning_obj: QuantileBinning = None
        self.summary_statistics = None
        self.header = None
        # self.quantile_summary_dict = {}
        self.cols_dict = {}
        # self.medians = None
        self.data_instances = data_instances
        self.cols_index = None
        if not isinstance(abnormal_list, list):
            abnormal_list = [abnormal_list]

        self.abnormal_list = abnormal_list
        self.__init_cols(data_instances, cols_index, stat_order, bias)
        self.label_summary = None
        self.error = error

    def __init_cols(self, data_instances, cols_index, stat_order, bias):
        header = data_overview.get_header(data_instances)
        self.header = header
        if cols_index == -1:
            self.cols_index = [i for i in range(len(header))]
        else:
            self.cols_index = cols_index
        LOGGER.debug(f"col_index: {cols_index}, self.col_index: {self.cols_index}")
        self.cols_dict = {header[indices]: indices for indices in self.cols_index}
        self.summary_statistics = SummaryStatistics(length=len(self.cols_index),
                                                    abnormal_list=self.abnormal_list,
                                                    stat_order=stat_order,
                                                    bias=bias)

    def _static_sums(self):
        """
        Statics sum, sum_square, max_value, min_value,
        so that variance is available.
        """
        is_sparse = data_overview.is_sparse_data(self.data_instances)
        partition_cal = functools.partial(self.static_in_partition,
                                          cols_index=self.cols_index,
                                          summary_statistics=copy.deepcopy(self.summary_statistics),
                                          is_sparse=is_sparse)
        self.summary_statistics = self.data_instances.applyPartitions(partition_cal). \
            reduce(lambda x, y: self.copy_merge(x, y))
        # self.summary_statistics = summary_statistic_dict.reduce(self.aggregate_statics)
        self.finish_fit_statics = True

    def _static_quantile_summaries(self):
        """
        Static summaries so that can query a specific quantile point
        """
        if self.binning_obj is not None:
            return self.binning_obj
        bin_param = FeatureBinningParam(bin_num=2, bin_indexes=self.cols_index,
                                        error=self.error)
        self.binning_obj = QuantileBinning(bin_param, abnormal_list=self.abnormal_list)
        self.binning_obj.fit_split_points(self.data_instances)

        return self.binning_obj

    @staticmethod
    def copy_merge(s1, s2):
        new_s1 = copy.deepcopy(s1)
        return new_s1.merge(s2)

    @staticmethod
    def static_in_partition(data_instances, cols_index, summary_statistics, is_sparse):
        """
        Statics sums, sum_square, max and min value through one traversal

        Parameters
        ----------
        data_instances : DTable
            The input data

        cols_index : indices
            Specify which column(s) need to apply statistic.

        summary_statistics: SummaryStatistics

        Returns
        -------
        Dict of SummaryStatistics object

        """

        for k, instances in data_instances:
            if not is_sparse:
                if isinstance(instances, Instance):
                    features = instances.features
                else:
                    try:
                        features = np.array(instances, dtype=float)
                    except ValueError as e:
                        raise ValueError(f"Static Module accept numeric input only. Error info: {e}")
                LOGGER.debug(f"In statics, features: {features}")
                row_values = features[cols_index]
            else:
                sparse_data = instances.features.get_sparse_vector()
                row_values = np.array([sparse_data.get(x, 0) for x in cols_index])
            summary_statistics.add_rows(row_values)
        return summary_statistics

    @staticmethod
    def static_summaries_in_partition(data_instances, cols_dict, abnormal_list, error):
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
            summary_dict[col_name] = QuantileSummaries(abnormal_list=abnormal_list, error=error)

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

    def get_median(self):
        if self.binning_obj is None:
            self._static_quantile_summaries()

        medians = self.binning_obj.query_quantile_point(query_points=0.5)
        return medians

    @property
    def median(self):
        median_dict = self.get_median()
        return np.array([median_dict[self.header[idx]] for idx in self.cols_index])

    def get_quantile_point(self, quantile):
        """
        Return the specific quantile point value

        Parameters
        ----------
        quantile : float, 0 <= quantile <= 1
            Specify which column(s) need to apply statistic.

        Returns
        -------
        return a dict of result quantile points.
        eg.
        quantile_point = {"x1": 3, "x2": 5... }
        """

        if self.binning_obj is None:
            self._static_quantile_summaries()
        quantile_points = self.binning_obj.query_quantile_point(quantile)
        return quantile_points

    def get_mean(self):
        """
        Return the mean value(s) of the given column

        Returns
        -------
        return a dict of result mean.

        """
        return self.get_statics("mean")

    def get_variance(self):
        return self.get_statics("variance")

    def get_std_variance(self):
        return self.get_statics("stddev")

    def get_max(self):
        return self.get_statics("max_value")

    def get_min(self):
        return self.get_statics("min_value")

    def get_statics(self, data_type):
        """
        Return the specific static value(s) of the given column

        Parameters
        ----------
        data_type : str, "mean", "variance", "std_variance", "max_value" or "mim_value"
            Specify which type to show.

        Returns
        -------
        return a list of result result. The order is the same as cols.
        """
        if not self.finish_fit_statics:
            self._static_sums()

        if hasattr(self.summary_statistics, data_type):
            result_row = getattr(self.summary_statistics, data_type)

        elif hasattr(self, data_type):
            result_row = getattr(self, data_type)
        else:
            raise ValueError(f"Statistic data type: {data_type} cannot be recognized")
        LOGGER.debug(f"col_index: {self.cols_index}, result_row: {result_row},"
                     f"header: {self.header}, data_type: {data_type}")
        # result = {self.header[header_idx]: result_row[col_idx]
        #           for col_idx, header_idx in enumerate(self.cols_index)}
        result = {}

        result_row = result_row.tolist()
        for col_idx, header_idx in enumerate(self.cols_index):
            result[self.header[header_idx]] = result_row[col_idx]
        return result

    def get_missing_ratio(self):
        return self.get_statics("missing_ratio")

    @property
    def missing_ratio(self):
        missing_static_obj = MissingStatistic()
        all_missing_ratio = missing_static_obj.fit(self.data_instances)
        return np.array([all_missing_ratio[self.header[idx]] for idx in self.cols_index])

    @property
    def missing_count(self):
        missing_ratio = self.missing_ratio
        missing_count = missing_ratio * self.data_instances.count()
        return missing_count.astype(int)

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
        label_histogram = self.data_instances.applyPartitions(self.get_label_static_dict).reduce(self.merge_result_dict)
        return label_histogram
