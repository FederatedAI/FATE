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
import numpy as np

from federatedml.util import consts, LOGGER

"""
Structure of compressed object, for memory saving we use tuple (value, g, delta) in fate>=v1.8
"""

"""
class Stats(object):
    def __init__(self, value, g: int, delta: int):
        self.value = value
        self.g = g
        self.delta = delta
"""


class QuantileSummaries(object):
    def __init__(self, compress_thres=consts.DEFAULT_COMPRESS_THRESHOLD,
                 head_size=consts.DEFAULT_HEAD_SIZE,
                 error=consts.DEFAULT_RELATIVE_ERROR,
                 abnormal_list=None):
        self.compress_thres = compress_thres
        self.head_size = head_size
        self.error = error
        self.head_sampled = []
        self.sampled = []  # list of Stats
        self.count = 0  # Total observations appeared
        self.missing_count = 0
        if abnormal_list is None:
            self.abnormal_list = []
        else:
            self.abnormal_list = abnormal_list

    # insert a number
    def insert(self, x):
        """
        Insert an observation of data. First store in a array buffer. If the buffer is full,
        do a batch insert. If the size of sampled list reach compress_thres, compress this list.
        Parameters
        ----------
        x : float
            The observation that prepare to insert

        """
        if x in self.abnormal_list or (isinstance(x, float) and np.isnan(x)):
            self.missing_count += 1
            return

        x = float(x)

        self.head_sampled.append(x)
        if len(self.head_sampled) >= self.head_size:
            self._insert_head_buffer()
            if len(self.sampled) >= self.compress_thres:
                self.compress()

    def _insert_head_buffer(self):
        if not len(self.head_sampled):  # If empty
            return
        current_count = self.count
        sorted_head = sorted(self.head_sampled)
        head_len = len(sorted_head)
        sample_len = len(self.sampled)
        new_sampled = []
        sample_idx = 0
        ops_idx = 0
        while ops_idx < head_len:
            current_sample = sorted_head[ops_idx]
            while sample_idx < sample_len and self.sampled[sample_idx][0] <= current_sample:
                new_sampled.append(self.sampled[sample_idx])
                sample_idx += 1

            current_count += 1

            # If it is the first one to insert or if it is the last one
            if not new_sampled or (sample_idx == sample_len and
                                   ops_idx == head_len - 1):
                delta = 0
            else:
                # delta = math.floor(2 * self.error * current_count) - 1
                delta = math.floor(2 * self.error * current_count)

            new_sampled.append((current_sample, 1, delta))
            ops_idx += 1

        new_sampled += self.sampled[sample_idx:]
        self.sampled = new_sampled
        self.head_sampled = []
        self.count = current_count

    def compress(self):
        self._insert_head_buffer()
        # merge_threshold = math.floor(2 * self.error * self.count) - 1
        merge_threshold = 2 * self.error * self.count
        compressed = self._compress_immut(merge_threshold)
        self.sampled = compressed

    def merge(self, other):
        """
        merge current summeries with the other one.
        Parameters
        ----------
        other : QuantileSummaries
            The summaries to be merged
        """
        if other.head_sampled:
            # other._insert_head_buffer()
            other.compress()

        if self.head_sampled:
            # self._insert_head_buffer()
            self.compress()

        if other.count == 0:
            return self

        if self.count == 0:
            return other

        # merge two sorted array
        new_sample = []
        i, j = 0, 0
        self_sample_len = len(self.sampled)
        other_sample_len = len(other.sampled)
        while i < self_sample_len and j < other_sample_len:
            if self.sampled[i][0] < other.sampled[j][0]:
                new_sample.append(self.sampled[i])
                i += 1
            else:
                new_sample.append(other.sampled[j])
                j += 1
        new_sample += self.sampled[i:]
        new_sample += other.sampled[j:]

        res_summary = self.__class__(compress_thres=self.compress_thres,
                                     head_size=self.head_size,
                                     error=self.error,
                                     abnormal_list=self.abnormal_list)
        res_summary.count = self.count + other.count
        res_summary.missing_count = self.missing_count + other.missing_count
        res_summary.sampled = new_sample
        # self.sampled = new_sample
        # self.count += other.count
        # merge_threshold = math.floor(2 * self.error * self.count) - 1
        merge_threshold = 2 * self.error * res_summary.count

        res_summary.sampled = res_summary._compress_immut(merge_threshold)
        return res_summary

    def query(self, quantile):
        """
        Given the queried quantile, return the approximation guaranteed result
        Parameters
        ----------
        quantile : float [0.0, 1.0]
            The target quantile

        Returns
        -------
        float, the corresponding value result.
        """
        if self.head_sampled:
            # self._insert_head_buffer()
            self.compress()

        if quantile < 0 or quantile > 1:
            raise ValueError("Quantile should be in range [0.0, 1.0]")

        if self.count == 0:
            return 0

        if quantile <= self.error:
            return self.sampled[0][0]

        if quantile >= 1 - self.error:
            return self.sampled[-1][0]

        rank = math.ceil(quantile * self.count)
        target_error = math.ceil(self.error * self.count)
        min_rank = 0
        i = 1
        while i < len(self.sampled) - 1:
            cur_sample = self.sampled[i]
            min_rank += cur_sample[1]
            max_rank = min_rank + cur_sample[2]
            if max_rank - target_error <= rank <= min_rank + target_error:
                return cur_sample[0]
            i += 1
        return self.sampled[-1][0]

    def query_percentile_rate_list(self, percentile_rate_list):
        if self.head_sampled:
            self.compress()

        if np.min(percentile_rate_list) < 0 or np.max(percentile_rate_list) > 1:
            raise ValueError("Quantile should be in range [0.0, 1.0]")

        if self.count == 0:
            return [0] * len(percentile_rate_list)

        split_points = []
        i, j = 0, len(percentile_rate_list) - 1
        while i < len(percentile_rate_list) and percentile_rate_list[i] <= self.error:
            split_points.append(self.sampled[0][0])
            # split_points.append(self.sampled[0].value)
            i += 1

        while j >= 0 and percentile_rate_list[i] >= 1 - self.error:
            j -= 1

        k = 1
        min_rank = 0
        while i <= j:
            quantile = percentile_rate_list[i]
            rank = math.ceil(quantile * self.count)
            target_error = math.ceil(self.error * self.count)
            while k < len(self.sampled) - 1:
                # cur_sample = self.sampled[k]
                # min_rank += cur_sample.g
                # max_rank = min_rank + cur_sample.delta
                cur_sample_value = self.sampled[k][0]
                min_rank += self.sampled[k][1]
                max_rank = min_rank + self.sampled[k][2]
                if max_rank - target_error <= rank <= min_rank + target_error:
                    split_points.append(cur_sample_value)
                    min_rank -= self.sampled[k][1]
                    break
                k += 1

            if k == len(self.sampled) - 1:
                # split_points.append(self.sampled[-1].value)
                split_points.append(self.sampled[-1][0])

            i += 1

        while j + 1 < len(percentile_rate_list):
            j += 1
            split_points.append(self.sampled[-1][0])

        assert len(percentile_rate_list) == len(split_points)
        return split_points

    def value_to_rank(self, value):
        min_rank, max_rank = 0, 0
        for sample in self.sampled:
            if sample[0] < value:
                min_rank += sample[1]
                max_rank = min_rank + sample[2]
            else:
                return (min_rank + max_rank) // 2
        return (min_rank + max_rank) // 2

    def query_value_list(self, values):
        """
        Given a sorted value list, return the rank of each element in this list
        """
        self.compress()
        res = []
        min_rank, max_rank = 0, 0
        idx = 0
        sample_idx = 0

        while sample_idx < len(self.sampled):
            v = values[idx]
            sample = self.sampled[sample_idx]
            if sample[0] <= v:
                min_rank += sample[1]
                max_rank = min_rank + sample[2]
                sample_idx += 1
            else:
                res.append((min_rank + max_rank) // 2)
                idx += 1
                if idx >= len(values):
                    break

        while idx < len(values):
            res.append((min_rank + max_rank) // 2)
            idx += 1
        return res

    def _compress_immut(self, merge_threshold):
        if not self.sampled:
            return self.sampled

        res = []

        # Start from the last element
        head = self.sampled[-1]
        sum_g_delta = head[1] + head[2]
        i = len(self.sampled) - 2  # Do not merge the last element

        while i >= 1:
            this_sample = self.sampled[i]
            if this_sample[1] + sum_g_delta < merge_threshold:
                head = (head[0], head[1] + this_sample[1], head[2])
                sum_g_delta += this_sample[1]
            else:
                res.append(head)
                head = this_sample
                sum_g_delta = head[1] + head[2]
            i -= 1
        res.append(head)

        # If head of current sample is smaller than this new res's head
        # Add current head into res
        current_head = self.sampled[0]
        if current_head[0] <= head[0] and len(self.sampled) > 1:
            res.append(current_head)

        # Python do not support prepend, thus, use reverse instead
        res.reverse()
        return res


class SparseQuantileSummaries(QuantileSummaries):
    def __init__(self, compress_thres=consts.DEFAULT_COMPRESS_THRESHOLD,
                 head_size=consts.DEFAULT_HEAD_SIZE,
                 error=consts.DEFAULT_RELATIVE_ERROR,
                 abnormal_list=None):
        super(SparseQuantileSummaries, self).__init__(compress_thres, head_size, error, abnormal_list)

        # Compare with the sparse point, static the number of each part.
        self.smaller_num = 0
        self.bigger_num = 0
        self._total_count = 0

    def set_total_count(self, total_count):
        self._total_count = total_count
        return self

    @property
    def summary_count(self):
        return self._total_count - self.missing_count

    def insert(self, x):
        if x in self.abnormal_list or np.isnan(x):
            self.missing_count += 1
            return
        if x < consts.FLOAT_ZERO:
            self.smaller_num += 1
        elif x >= consts.FLOAT_ZERO:
            self.bigger_num += 1
        super(SparseQuantileSummaries, self).insert(x)

    def query(self, quantile):
        if self.zero_lower_bound < quantile < self.zero_upper_bound:
            return 0.0

        non_zero_quantile = self._convert_query_percentile(quantile)
        result = super(SparseQuantileSummaries, self).query(non_zero_quantile)
        return result

    def query_percentile_rate_list(self, percentile_rate_list):
        result = []
        non_zero_quantile_list = list()
        for quantile in percentile_rate_list:
            if self.zero_lower_bound < quantile < self.zero_upper_bound:
                result.append(0.0)
            else:
                non_zero_quantile_list.append(self._convert_query_percentile(quantile))

        if non_zero_quantile_list:
            result += super(SparseQuantileSummaries, self).query_percentile_rate_list(non_zero_quantile_list)

        return result

    def value_to_rank(self, value):
        quantile_rank = super().value_to_rank(value)
        zeros_count = self.zero_counts

        if value > 0:
            return quantile_rank + zeros_count
        elif value < 0:
            return quantile_rank
        else:
            return quantile_rank + zeros_count // 2

    def merge(self, other):
        assert isinstance(other, SparseQuantileSummaries)
        res_summary = super(SparseQuantileSummaries, self).merge(other)
        res_summary.smaller_num = self.smaller_num + other.smaller_num
        res_summary.bigger_num = self.bigger_num + other.bigger_num
        return res_summary

    def _convert_query_percentile(self, quantile):
        zeros_count = self.zero_counts
        if zeros_count == 0:
            return quantile

        if quantile <= self.zero_lower_bound:
            return ((self._total_count - self.missing_count) / self.count) * quantile

        return (quantile - self.zero_upper_bound + self.zero_lower_bound) / (
            1 - self.zero_upper_bound + self.zero_lower_bound)

    @property
    def zero_lower_bound(self):
        if self.smaller_num == 0:
            return 0.0
        return self.smaller_num / (self._total_count - self.missing_count)

    @property
    def zero_upper_bound(self):
        if self.bigger_num == 0:
            return self._total_count - self.missing_count
        return (self.smaller_num + self.zero_counts) / (self._total_count - self.missing_count)

    @property
    def zero_counts(self):
        return self._total_count - self.smaller_num - self.bigger_num - self.missing_count

    def query_value_list(self, values):
        summary_ranks = super().query_value_list(values)
        res = []
        for v, r in zip(values, summary_ranks):
            if v == 0:
                res.append(self.smaller_num)
            elif v < 0:
                res.append(r)
            else:
                res.append(r + self.zero_counts)
        return res


def quantile_summary_factory(is_sparse, param_dict):
    if is_sparse:
        return SparseQuantileSummaries(**param_dict)
    else:
        return QuantileSummaries(**param_dict)
