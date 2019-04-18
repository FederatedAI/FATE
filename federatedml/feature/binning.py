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

import functools
import math

import numpy as np

from arch.api.utils import log_utils
from federatedml.feature.quantile_summaries import QuantileSummaries
from federatedml.statistic.data_overview import get_features_shape

LOGGER = log_utils.getLogger()


class IVAttributes(object):
    def __init__(self, woe_array, iv_array, event_count_array, non_event_count_array,
                 event_rate_array, non_event_rate_array, split_points, iv=None):
        self.woe_array = woe_array
        self.iv_array = iv_array
        self.event_count_array = event_count_array
        self.non_event_count_array = non_event_count_array
        self.event_rate_array = event_rate_array
        self.non_event_rate_array = non_event_rate_array
        if split_points is None:
            self.split_points = []
        else:
            self.split_points = []
            # Remove those repeated split points
            for s_p in split_points:
                if s_p not in self.split_points:
                    self.split_points.append(s_p)
        if iv is None:
            iv = 0
            for idx, woe in enumerate(self.woe_array):
                non_event_rate = non_event_count_array[idx]
                event_rate = event_rate_array[idx]
                iv += (non_event_rate - event_rate) * woe
        self.iv = iv

    @property
    def is_woe_monotonic(self):
        """
        Check the woe is monotonic or not
        """
        woe_array = self.woe_array
        if len(woe_array) <= 1:
            return True

        is_increasing = all(x <= y for x, y in zip(woe_array, woe_array[1:]))
        is_decreasing = all(x >= y for x, y in zip(woe_array, woe_array[1:]))
        return is_increasing or is_decreasing

    @property
    def bin_nums(self):
        return len(self.woe_array)

    def result_dict(self):
        save_dict = self.__dict__
        save_dict['is_woe_monotonic'] = self.is_woe_monotonic
        save_dict['bin_nums'] = self.bin_nums
        return save_dict

    def display_result(self, display_results):
        save_dict = self.result_dict()
        dis_str = ""
        for d_s in display_results:
            dis_str += "{} is {};\n".format(d_s, save_dict.get(d_s))
        return dis_str

    def reconstruct(self, iv_obj):
        self.woe_array = list(iv_obj.woe_array)
        self.iv_array = list(iv_obj.iv_array)
        self.event_count_array = list(iv_obj.event_count_array)
        self.non_event_count_array = list(iv_obj.non_event_count_array)
        self.event_rate_array = list(iv_obj.event_rate_array)
        self.non_event_rate_array = list(iv_obj.non_event_rate_array)
        self.split_points = list(iv_obj.split_points)
        self.iv = iv_obj.iv

class Binning(object):
    def __init__(self, params):
        self.params = params
        self.bin_num = params.bin_num

    def binning(self, data_instances, cols):
        raise NotImplementedError("Should not call this class directly")

    def transform(self, data_instances, split_points=None, cols=-1):
        """
        Apply the binning method

        Parameters
        ----------
        data_instances : DTable
            The input data

        split_points : list.
            Each row represent for the split points for a feature. The element in each row represent for
            the corresponding split point.
            e.g.
            split_points = [[0.1, 0.2, 0.3, 0.4 ...],    # The first feature
                            [1, 2, 3, 4, ...],           # The second feature
                            ...]                         # Other features

        cols : int or list of int
            Specify which column(s) need to apply binning. -1 means do binning for all columns.

        Returns
        -------
        data_bin_table : DTable.
            The element in each row represent for the corresponding bin number this feature belongs to.
            e.g. for each row, it could be:
            (1, 5, 2, 6, 0, ...)    # Each number represent for the bin number it belongs to. The order is the
                                # same as the order of cols.


        """
        if cols == -1:
            features_shape = get_features_shape(data_instances)
            if features_shape is None:
                raise RuntimeError('Cannot get feature shape, please check input data')
            cols = [i for i in range(features_shape)]

        if isinstance(cols, int):
            cols = [cols]

        assert len(split_points) == len(cols)

        if split_points is None:
            split_points = self.binning(data_instances, cols)

        f = functools.partial(self.bin_data,
                              split_points=split_points,
                              cols=cols)
        data_bin_table = data_instances.mapValues(f)
        return data_bin_table

    def cal_local_iv(self, data_instances, cols, split_points=None, label_table=None):
        if cols == -1:
            features_shape = get_features_shape(data_instances)
            if features_shape is None:
                raise RuntimeError('Cannot get feature shape, please check input data')
            cols = [i for i in range(features_shape)]

        if split_points is None:
            split_points = self.binning(data_instances, cols=cols)

        data_bin_table = self.transform(data_instances, split_points, cols)
        if label_table is None:
            label_table = data_instances.mapValues(lambda x: x.label)
        event_count_table = label_table.mapValues(lambda x: (x, 1 - x))
        data_bin_with_label = data_bin_table.join(event_count_table, lambda x, y: (x, y))
        f = functools.partial(self.add_label_in_partition,
                              total_bin=self.bin_num,
                              cols=cols)

        result_sum = data_bin_with_label.mapPartitions(f)
        result_counts = result_sum.reduce(self.aggregate_partition_label)

        iv_attrs = self.cal_iv_woe(result_counts, self.params.adjustment_factor,
                                   split_points=split_points)
        return iv_attrs

    @staticmethod
    def bin_data(instance, split_points, cols):
        result_bin_nums = []
        for col_index, col in enumerate(cols):
            col_split_points = split_points[col_index]

            value = instance.features[col]
            col_bin_num = len(col_split_points)
            for bin_num, split_point in enumerate(col_split_points):
                if value < split_point:
                    col_bin_num = bin_num
                    break
            result_bin_nums.append(col_bin_num)
        result_bin_nums = tuple(result_bin_nums)
        return result_bin_nums

    @staticmethod
    def woe_1d(data_event_count, adjustment_factor, split_points):
        """
        Given event and non-event count in one column, calculate its woe value.

        Parameters
        ----------
        data_event_count : list
            [(event_sum, non-event_sum), (same sum in second_bin), (in third bin) ...]

        adjustment_factor : float
            The adjustment factor when calculating WOE

        split_points : list
            Use to display in the final result

        Returns
        -------
        IVAttributes : object
            Stored information that related iv and woe value
        """
        event_total = 0
        non_event_total = 0
        for event_sum, non_event_sum in data_event_count:
            event_total += event_sum
            non_event_total += non_event_sum
        # LOGGER.debug("In woe_1d func, data_event_count is {}, event_total: {}, non_event_total: {}".format(
        #     data_event_count, event_total, non_event_total
        # ))
        if event_total == 0:
            raise ValueError("NO event label in target data")
        if non_event_total == 0:
            raise ValueError("NO non-event label in target data")

        iv = 0
        event_count_array = []
        non_event_count_array = []
        event_rate_array = []
        non_event_rate_array = []
        woe_array = []
        iv_array = []

        for event_count, non_event_count in data_event_count:
            if event_count == 0 and non_event_count == 0:
                continue
            if event_count == 0 or non_event_count == 0:
                event_rate = 1.0 * (event_count + adjustment_factor) / event_total
                non_event_rate = 1.0 * (non_event_count + adjustment_factor) / non_event_total
            else:
                event_rate = 1.0 * event_count / event_total
                non_event_rate = 1.0 * non_event_count / non_event_total
            woe_i = math.log(non_event_rate / event_rate)

            event_count_array.append(event_count)
            non_event_count_array.append(non_event_count)
            event_rate_array.append(event_rate)
            non_event_rate_array.append(non_event_rate)
            woe_array.append(woe_i)
            iv_i = (non_event_rate - event_rate) * woe_i
            iv_array.append(iv_i)
            iv += iv_i
        return IVAttributes(woe_array=woe_array, iv_array=iv_array, event_count_array=event_count_array,
                            non_event_count_array=non_event_count_array, split_points=split_points,
                            event_rate_array=event_rate_array, non_event_rate_array=non_event_rate_array, iv=iv)

    @staticmethod
    def cal_iv_woe(result_counts, adjustment_factor, split_points=None):
        """
        Given event count information calculate iv information

        Parameters
        ----------
        result_counts: DTable.
            It is like:
            [[(event_sum, non-event_sum), (same sum in second_bin), (in third bin) ... (first col in cols)],
            [(event_sum, non-event_sum), ... (second col in cols)],
            ...]

        adjustment_factor : float
            The adjustment factor when calculating WOE

        split_points : list
            Use to display in the final result

        Returns
        -------
        list of IVAttributes object
        """
        result_ivs = []
        for idx, data_event_count in enumerate(result_counts):
            if split_points is not None:
                feature_split_point = split_points[idx]
            else:
                feature_split_point = None
            result_ivs.append(Binning.woe_1d(data_event_count, adjustment_factor, feature_split_point))
        return result_ivs

    @staticmethod
    def add_label_in_partition(data_bin_with_table, total_bin, cols, encryptor=None):
        """
        Add all label, so that become convenient to calculate woe and iv

        Parameters
        ----------
        data_bin_with_table : DTable
            The input data, the DTable is like:
            (id, (1, 5, 2, 6, 0, ...), y, 1 - y)

        total_bin : int, > 0
            Specify the largest bin number

        cols : list of int or int,
            Specify which columns need to calculated. -1 represent for all columns

        Returns
        -------
        result_sum: the result DTable. It is like:
            [[(event_count, non_event_count), (same sum in second_bin), (in third bin) ... (first feature in cols)],
             [(event_count, non_event_count), ... (second feature in cols)],
             ...]

        """
        result_sum = []
        # LOGGER.debug("In add_label_in_partition, cols: {}".format(cols))
        for _ in cols:
            result_col_sum = []
            for bin_index in range(total_bin):
                if encryptor is not None:
                    result_col_sum.append([encryptor.encrypt(0), encryptor.encrypt(0)])
                else:
                    result_col_sum.append([0, 0])
            result_sum.append(result_col_sum)

        for _, datas in data_bin_with_table:
            bin_idxs = datas[0]
            y_combo = datas[1]
            # LOGGER.debug("In data_bin_with_table loop, bin_idxs: {}, y: {}, inverse_y: {}".format(
            #     bin_idxs, y_combo[0], y_combo[1]
            # ))
            y = y_combo[0]
            inverse_y = y_combo[1]
            for col_idx, bin_idx in enumerate(bin_idxs):
                col_sum = result_sum[col_idx]
                label_sum = col_sum[bin_idx]
                label_sum[0] = label_sum[0] + y
                label_sum[1] = label_sum[1] + inverse_y
                col_sum[bin_idx] = label_sum
                result_sum[col_idx] = col_sum

        return result_sum

    @staticmethod
    def aggregate_partition_label(sum1, sum2):
        """
        Used in reduce function. Aggregate the result calculate from each partition.

        Parameters
        ----------
        sum1 :  DTable.
            It is like:
            [[(event_count, non_event_count), (same sum in second_bin), (in third bin) ... (first feature in cols)],
            [(event_count, non_event_count), ... (second feature in cols)],
            ...]

        sum2 : DTable
            Same as sum1
        Returns
        -------
        Merged sum. The format is same as sum1.

        """
        if sum1 is None and sum2 is None:
            return None

        if sum1 is None:
            return sum2

        if sum2 is None:
            return sum1

        new_result = []
        for idx, feature_sum1 in enumerate(sum1):
            feature_sum2 = sum2[idx]
            tmp_list = []
            for idx, label_sum1 in enumerate(feature_sum1):
                label_sum2 = feature_sum2[idx]
                tmp = (label_sum1[0] + label_sum2[0], label_sum1[1] + label_sum2[1])
                tmp_list.append(tmp)
            new_result.append(tmp_list)
        return new_result


class QuantileBinning(Binning):
    """
    After quantile binning, the numbers of elements in each binning are equal.

    The result of this algorithm has the following deterministic bound:
    If the data_instances has N elements and if we request the quantile at probability `p` up to error
    `err`, then the algorithm will return a sample `x` from the data so that the *exact* rank
    of `x` is close to (p * N).
    More precisely,

    {{{
      floor((p - 2 * err) * N) <= rank(x) <= ceil((p + 2 * err) * N)
    }}}

    This method implements a variation of the Greenwald-Khanna algorithm (with some speed
    optimizations).
    """

    def __init__(self, params):
        super(QuantileBinning, self).__init__(params)
        self.summary_list = None

    def binning(self, data_instances, cols):
        """
        Apply the binning method

        Parameters
        ----------
        data_instances : DTable
            The input data

        cols : int or list of int
            Specify which column(s) need to apply binning. -1 means do binning for all columns.

        Returns
        -------
        split_points, 2-dimension list.
            Each row represent for the split points for a feature. The element in each row represent for
            the corresponding split point.
            e.g.
            split_points = [[0.1, 0.2, 0.3, 0.4 ...],    # The first feature
                            [1, 2, 3, 4, ...],           # The second feature
                            ...]                         # Other features

        """
        percent_value = 1.0 / self.bin_num
        # calculate the split points
        percentile_rate = [i * percent_value for i in range(1, self.bin_num)]

        if self.summary_list is None:
            f = functools.partial(self.approxiQuantile,
                                  cols=cols,
                                  params=self.params)
            summary_list = data_instances.mapPartitions(f)
            summary_list = summary_list.reduce(self.merge_summary_list)
            self.summary_list = summary_list
        else:
            summary_list = self.summary_list
        split_points = []
        for percen_rate in percentile_rate:
            feature_dimension_points = [s_l.query(percen_rate) for s_l in summary_list]
            split_points.append(feature_dimension_points)
        split_points = np.array(split_points)
        split_points = split_points.transpose()
        return split_points

    @staticmethod
    def approxiQuantile(data_instances, cols, params):
        # cols == -1 means all features
        if cols == -1:
            features_shape = get_features_shape(data_instances)
            if features_shape is None:
                raise RuntimeError('Cannot get feature shape, please check input data')
            cols = [i for i in range(features_shape)]

        if isinstance(cols, int):
            cols = [cols]

        num_of_qs = len(cols)
        summary_list = []
        for _ in range(num_of_qs):
            quantile_summaries = QuantileSummaries(compress_thres=params.compress_thres,
                                                   head_size=params.head_size,
                                                   error=params.error)
            summary_list.append(quantile_summaries)
        QuantileBinning.insert_datas(data_instances, summary_list, cols)
        return summary_list

    @staticmethod
    def insert_datas(data_instances, summary_list, cols):

        for iter_key, instant in data_instances:
            features = instant.features
            for idx, summary in enumerate(summary_list):
                feature_id = cols[idx]
                summary.insert(features[feature_id])

    @staticmethod
    def merge_summary_list(s_list1, s_list2):
        if s_list1 is None and s_list2 is None:
            return None
        if s_list1 is None:
            return s_list2
        if s_list2 is None:
            return s_list1

        new_list = []
        for idx, summary1 in enumerate(s_list1):
            summary1.merge(s_list2[idx])
            new_list.append(summary1)
        return new_list

    def query_quantile_point(self, data_instances, cols, query_points):
        if self.summary_list is None:
            f = functools.partial(self.approxiQuantile,
                                  cols=cols,
                                  params=self.params)
            summary_list = data_instances.mapPartitions(f)
            summary_list = summary_list.reduce(self.merge_summary_list)
            self.summary_list = summary_list
        else:
            summary_list = self.summary_list

        if isinstance(query_points, (int, float)):
            query_points = [query_points] * len(cols)

        if len(cols) != len(query_points) or len(summary_list) != len(query_points):
            raise AssertionError("number of quantile points are not equal to number of select_cols")

        result = []
        for idx, query_point in enumerate(query_points):
            summary = summary_list[idx]
            result.append(summary.query(query_point))

        return result


