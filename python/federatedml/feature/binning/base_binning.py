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

import bisect
import copy
import functools
import math
import random
import operator

import numpy as np

from federatedml.feature.binning.bin_inner_param import BinInnerParam
from federatedml.feature.binning.bin_result import BinColResults, BinResults
from federatedml.feature.sparse_vector import SparseVector
from federatedml.statistic import data_overview
from federatedml.util import LOGGER


# from federatedml.statistic import statics


class BaseBinning(object):
    """
    This is use for discrete data so that can transform data or use information for feature selection.
    """

    def __init__(self, params=None, abnormal_list=None):
        self.bin_inner_param: BinInnerParam = None
        self.bin_results = BinResults()
        if params is None:
            return
        self.params = params
        self.bin_num = params.bin_num
        self.event_total = None
        self.non_event_total = None
        self.bin_data_result = None

        if abnormal_list is None:
            self.abnormal_list = []
        else:
            self.abnormal_list = abnormal_list

    def set_role_party(self, role, party_id):
        self.bin_results.set_role_party(role, party_id)

    @property
    def header(self):
        return self.bin_inner_param.header

    @property
    def split_points(self):
        return self.bin_results.all_split_points

    def _default_setting(self, header):
        if self.bin_inner_param is not None:
            return
        bin_inner_param = BinInnerParam()
        bin_inner_param.set_header(header)
        bin_inner_param.set_bin_all()
        bin_inner_param.set_transform_all()
        self.set_bin_inner_param(bin_inner_param)

    def fit_split_points(self, data_instances):
        """
        Get split points

        Parameters
        ----------
        data_instances : DTable
            The input data

        Returns
        -------

        split_points : dict.
            Each value represent for the split points for a feature. The element in each row represent for
            the corresponding split point.
            e.g.
            split_points = {'x1': [0.1, 0.2, 0.3, 0.4 ...],    # The first feature
                            'x2': [1, 2, 3, 4, ...],           # The second feature
                            ...]                         # Other features

        """
        pass

    def fit_category_features(self, data_instances):
        is_sparse = data_overview.is_sparse_data(data_instances)

        if len(self.bin_inner_param.category_indexes) > 0:
            statics_obj = data_overview.DataStatistics()
            category_col_values = statics_obj.static_all_values(data_instances,
                                                                self.bin_inner_param.category_indexes,
                                                                is_sparse)
            for col_name, split_points in zip(self.bin_inner_param.category_names, category_col_values):
                self.bin_results.put_col_split_points(col_name, split_points)

    def set_bin_inner_param(self, bin_inner_param):
        self.bin_inner_param = bin_inner_param

    def transform(self, data_instances, transform_type):
        # self._init_cols(data_instances)
        for col_name in self.bin_inner_param.transform_bin_names:
            if col_name not in self.header:
                raise ValueError("Transform col_name: {} is not existed".format(col_name))

        if transform_type == 'bin_num':
            data_instances, _, _ = self.convert_feature_to_bin(data_instances)
        elif transform_type == 'woe':
            data_instances = self.convert_feature_to_woe(data_instances)

        return data_instances

    def convert_feature_to_woe(self, data_instances):
        is_sparse = data_overview.is_sparse_data(data_instances)
        schema = data_instances.schema

        if is_sparse:
            f = functools.partial(self._convert_sparse_data,
                                  bin_inner_param=self.bin_inner_param,
                                  bin_results=self.bin_results,
                                  abnormal_list=self.abnormal_list,
                                  convert_type='woe'
                                  )
            new_data = data_instances.mapValues(f)
        else:
            f = functools.partial(self._convert_dense_data,
                                  bin_inner_param=self.bin_inner_param,
                                  bin_results=self.bin_results,
                                  abnormal_list=self.abnormal_list,
                                  convert_type='woe')
            new_data = data_instances.mapValues(f)
        new_data.schema = schema
        return new_data

    def convert_feature_to_bin(self, data_instances, split_points=None, bin_inner_param=None):

        if bin_inner_param is None:
            bin_inner_param = self.bin_inner_param

        if self.bin_data_result is not None and \
                bin_inner_param.transform_bin_indexes == self.bin_inner_param.transform_bin_indexes:
            return self.bin_data_result

        is_sparse = data_overview.is_sparse_data(data_instances)
        schema = data_instances.schema

        if split_points is None:
            split_points = self.bin_results.all_split_points
        else:
            for col_name, sp in split_points.items():
                self.bin_results.put_col_split_points(col_name, sp)

        if is_sparse:
            f = functools.partial(self._convert_sparse_data,
                                  bin_inner_param=bin_inner_param,
                                  bin_results=self.bin_results,
                                  abnormal_list=self.abnormal_list,
                                  convert_type='bin_num'
                                  )
            new_data = data_instances.mapValues(f)
        else:
            f = functools.partial(self._convert_dense_data,
                                  bin_inner_param=bin_inner_param,
                                  bin_results=self.bin_results,
                                  abnormal_list=self.abnormal_list,
                                  convert_type='bin_num')
            new_data = data_instances.mapValues(f)
        new_data.schema = schema
        bin_sparse = self.get_sparse_bin(bin_inner_param.transform_bin_indexes, split_points)
        split_points_result = self.bin_results.get_split_points_array(bin_inner_param.transform_bin_names)
        self.bin_data_result = (new_data, split_points_result, bin_sparse)
        return self.bin_data_result

    @staticmethod
    def _convert_sparse_data(instances, bin_inner_param: BinInnerParam, bin_results: BinResults,
                             abnormal_list: list, convert_type: str = 'bin_num'):
        instances = copy.deepcopy(instances)
        all_data = instances.features.get_all_data()
        data_shape = instances.features.get_shape()
        indice = []
        sparse_value = []
        transform_cols_idx = bin_inner_param.transform_bin_indexes
        split_points_dict = bin_results.all_split_points

        for col_idx, col_value in all_data:
            if col_idx in transform_cols_idx:
                if col_value in abnormal_list:
                    indice.append(col_idx)
                    sparse_value.append(col_value)
                    continue
                # Maybe it is because missing value add in sparse value, but
                col_name = bin_inner_param.header[col_idx]
                split_points = split_points_dict[col_name]
                bin_num = BaseBinning.get_bin_num(col_value, split_points)
                indice.append(col_idx)
                if convert_type == 'bin_num':
                    sparse_value.append(bin_num)
                elif convert_type == 'woe':
                    col_results = bin_results.all_cols_results.get(col_name)
                    woe_value = col_results.woe_array[bin_num]
                    sparse_value.append(woe_value)
                else:
                    sparse_value.append(col_value)
            else:
                indice.append(col_idx)
                sparse_value.append(col_value)

        sparse_vector = SparseVector(indice, sparse_value, data_shape)
        instances.features = sparse_vector
        return instances

    def get_sparse_bin(self, transform_cols_idx, split_points_dict):
        """
        Get which bins the 0 located at for each column.
        """
        result = {}
        for col_idx in transform_cols_idx:
            col_name = self.header[col_idx]
            split_points = split_points_dict[col_name]
            sparse_bin_num = self.get_bin_num(0, split_points)
            result[col_idx] = sparse_bin_num
        return result

    @staticmethod
    def _convert_dense_data(instances, bin_inner_param: BinInnerParam, bin_results: BinResults,
                            abnormal_list: list, convert_type: str = 'bin_num'):
        instances = copy.deepcopy(instances)
        features = instances.features
        transform_cols_idx = bin_inner_param.transform_bin_indexes
        split_points_dict = bin_results.all_split_points

        for col_idx, col_value in enumerate(features):
            if col_idx in transform_cols_idx:
                if col_value in abnormal_list:
                    features[col_idx] = col_value
                    continue
                col_name = bin_inner_param.header[col_idx]
                split_points = split_points_dict[col_name]
                bin_num = BaseBinning.get_bin_num(col_value, split_points)
                if convert_type == 'bin_num':
                    features[col_idx] = bin_num
                elif convert_type == 'woe':
                    col_results = bin_results.all_cols_results.get(col_name)
                    woe_value = col_results.woe_array[bin_num]
                    features[col_idx] = woe_value
                else:
                    features[col_idx] = col_value

        instances.features = features
        return instances

    def cal_iv_by_bin_table(self, bin_data_table, sparse_bin_num):
        """
        Calculate iv attributes

        Parameters
        ----------
        bin_data_table : DTable
            The input data whose features are bin nums

        sparse_bin_num: dict
            The bin num of value 0.

        Returns
        -------
        result_counts: dict.
        It is like:
            {'x1': [[event_count, non_event_count], [event_count, non_event_count] ... ],
             'x2': [[event_count, non_event_count], [event_count, non_event_count] ... ],
             ...
            }


        Dict of IVAttributes object
        """
        result_counts = self.static_bin_label(bin_data_table, sparse_bin_num)
        self.cal_iv_woe(result_counts,
                        self.params.adjustment_factor)

    def static_bin_label(self, bin_data_table, sparse_bin_num):
        is_sparse = data_overview.is_sparse_data(bin_data_table)
        event_total = bin_data_table.mapValues(lambda x: x.label).reduce(operator.add)
        non_event_total = bin_data_table.count() - event_total

        max_bin_num = 0
        split_points = self.split_points
        for _, sp in split_points.items():
            if len(sp) > max_bin_num:
                max_bin_num = len(sp)

        bin_indexes = self.bin_inner_param.bin_indexes
        f = functools.partial(self.add_label_in_partition,
                              sparse_bin_num=sparse_bin_num,
                              shape=(len(bin_indexes), max_bin_num),
                              bin_indexes=bin_indexes,
                              is_sparse=is_sparse)
        label_array, bin_count_array = bin_data_table.applyPartitions(f).reduce(lambda x, y: (x[0] + y[0],
                                                                  x[1] + y[1]))
        non_event_array = bin_count_array - label_array
        one_total = np.sum(label_array, axis=1)
        zero_total = np.sum(non_event_array, axis=1)
        result_counts = {}
        for i, col_idx in enumerate(bin_indexes):
            col_name = self.bin_inner_param.bin_names[i]
            sps_len = len(split_points[col_name])
            _counts = []

            for j in range(sps_len):
                _counts.append((label_array[i][j], non_event_array[i][j]))
            _counts[sparse_bin_num[col_idx]] = (event_total - one_total[i], non_event_total - zero_total[i])
            result_counts[col_name] = _counts
        LOGGER.debug(f"tmc2, result_count: {result_counts}")
        return result_counts

    @staticmethod
    def add_label_in_partition(bin_data_table, sparse_bin_num, shape, bin_indexes, is_sparse):
        """
        Add all label, so that become convenient to calculate woe and iv

        Parameters
        ----------
        bin_data_table : DTable
            The input data whose features are bin_num.

        sparse_bin_num: dict
            which bins the 0 located at for each column.

        shape: tuple
            (len(bin_indexes), max_bin_num)

        bin_indexes: list
            List of which columns should be static

        is_sparse: bool
            Whether it is a sparse data input

        Returns
        -------
        result_sum: the result DTable. It is like:
            {'x1': [[event_count, total_num], [event_count, total_num] ... ],
             'x2': [[event_count, total_num], [event_count, total_num] ... ],
             ...
            }

        """
        label_array = np.zeros(shape=shape, dtype=object)
        bin_count_array = np.zeros(shape=shape, dtype=object)
        for _, instance in bin_data_table:
            if is_sparse:
                data_iter = instance.features.get_all_data()
            else:
                data_iter = enumerate(instance.features)
            for col_idx, bin_num in data_iter:
                bin_num = int(bin_num)
                if col_idx not in bin_indexes:
                    continue
                if bin_num == sparse_bin_num[col_idx]:
                    continue
                else:
                    LOGGER.debug(f"tmc: bin_num: {bin_num}, sparse_bin_num: {sparse_bin_num[col_idx]},"
                                 f"col_idx: {col_idx}")
                i = bin_indexes.index(col_idx)
                label_array[i][bin_num] = label_array[i][bin_num] + instance.label
                bin_count_array[i][bin_num] = bin_count_array[i][bin_num] + 1
        return label_array, bin_count_array

    @staticmethod
    def bin_data(instance, split_points, cols_dict, header, is_sparse):
        """
        Apply the binning method

        Parameters
        ----------
        instance : DTable
            The input data

        split_points : dict.
            Each value represent for the split points for a feature. The element in each row represent for
            the corresponding split point.
            e.g.
            split_points = {'x1': [0.1, 0.2, 0.3, 0.4 ...],    # The first feature
                            'x2': [1, 2, 3, 4, ...],           # The second feature
                            ...]                         # Other features

        cols_dict: dict
            Record key, value pairs where key is cols' name, and value is cols' index.

        is_sparse: bool
            Specify whether it is sparse data or not

        Returns
        -------
        result_bin_dict : dict.
            Each element represent for the corresponding bin number this feature belongs to.
            e.g. it could be:
            [{1: 1, 2: 5, 3: 2}
            ...
             ]  # Each number represent for the bin number it belongs to.
        """

        result_bin_nums = {}

        if is_sparse:
            sparse_data = instance.features.get_all_data()
            for col_idx, col_value in sparse_data:
                col_name = header[col_idx]

                if col_name in cols_dict:
                    col_split_points = split_points[col_name]
                    col_bin_num = BaseBinning.get_bin_num(col_value, col_split_points)
                    result_bin_nums[col_name] = col_bin_num
            return result_bin_nums

        # For dense data
        for col_name, col_index in cols_dict.items():
            col_split_points = split_points[col_name]

            value = instance.features[col_index]
            col_bin_num = BaseBinning.get_bin_num(value, col_split_points)
            result_bin_nums[col_name] = col_bin_num

        return result_bin_nums

    @staticmethod
    def get_bin_num(value, split_points):
        sp = split_points[:-1]
        col_bin_num = bisect.bisect_left(sp, value)
        # col_bin_num = bisect.bisect_left(split_points, value)
        return int(col_bin_num)

    @staticmethod
    def woe_1d(data_event_count, adjustment_factor):
        """
        Given event and non-event count in one column, calculate its woe value.

        Parameters
        ----------
        data_event_count : list
            [(event_sum, non-event_sum), (same sum in second_bin), (in third bin) ...]

        adjustment_factor : float
            The adjustment factor when calculating WOE

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

        if event_total == 0:
            # raise ValueError("NO event label in target data")
            event_total = 1
        if non_event_total == 0:
            # raise ValueError("NO non-event label in target data")
            non_event_total = 1

        iv = 0
        event_count_array = []
        non_event_count_array = []
        event_rate_array = []
        non_event_rate_array = []
        woe_array = []
        iv_array = []

        for event_count, non_event_count in data_event_count:

            if event_count == 0 or non_event_count == 0:
                event_rate = 1.0 * (event_count + adjustment_factor) / event_total
                non_event_rate = 1.0 * (non_event_count + adjustment_factor) / non_event_total
            else:
                event_rate = 1.0 * event_count / event_total
                non_event_rate = 1.0 * non_event_count / non_event_total
            woe_i = math.log(event_rate / non_event_rate)

            event_count_array.append(event_count)
            non_event_count_array.append(non_event_count)
            event_rate_array.append(event_rate)
            non_event_rate_array.append(non_event_rate)
            woe_array.append(woe_i)
            iv_i = (event_rate - non_event_rate) * woe_i
            iv_array.append(iv_i)
            iv += iv_i
        return BinColResults(woe_array=woe_array, iv_array=iv_array, event_count_array=event_count_array,
                             non_event_count_array=non_event_count_array,
                             event_rate_array=event_rate_array, non_event_rate_array=non_event_rate_array, iv=iv)

    def cal_iv_woe(self, result_counts, adjustment_factor):
        """
        Given event count information calculate iv information

        Parameters
        ----------
        result_counts: dict.
            It is like:
                {'x1': [[event_count, non_event_count], [event_count, non_event_count] ... ],
                 'x2': [[event_count, non_event_count], [event_count, non_event_count] ... ],
                 ...
                }

        adjustment_factor : float
            The adjustment factor when calculating WOE

        Returns
        -------
        Dict of IVAttributes object
            {'x1': attr_obj,
             'x2': attr_obj
             ...
             }
        """
        for col_name, data_event_count in result_counts.items():
            col_result_obj = self.woe_1d(data_event_count, adjustment_factor)
            assert isinstance(col_result_obj, BinColResults)
            self.bin_results.put_col_results(col_name, col_result_obj)

    def shuffle_static_counts(self, statistic_counts):
        """
        Shuffle bin orders, and stored orders in self.bin_results

        Parameters
        ----------
        statistic_counts :  dict.
            It is like:
                {'x1': [[event_count, non_event_count], [event_count, non_event_count] ... ],
                 'x2': [[event_count, non_event_count], [event_count, non_event_count] ... ],
                 ...
                }

        Returns
        -------
        Shuffled result
        """
        result = {}
        for col_name, count_sum in statistic_counts.items():
            this_bin_result = self.bin_results.all_cols_results.get(col_name, BinColResults())
            shuffled_index = [x for x in range(len(count_sum))]
            random.shuffle(shuffled_index)
            result[col_name] = [count_sum[i] for i in shuffled_index]
            this_bin_result.bin_anonymous = ["bin_" + str(i) for i in shuffled_index]
            self.bin_results.all_cols_results[col_name] = this_bin_result
        return result

    @staticmethod
    def aggregate_partition_label(sum1, sum2):
        """
        Used in reduce function. Aggregate the result calculate from each partition.

        Parameters
        ----------
        sum1 :  dict.
            It is like:
                {'x1': [[event_count, total_num], [event_count, total_num] ... ],
                 'x2': [[event_count, total_num], [event_count, total_num] ... ],
                 ...
                }

        sum2 : dict
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

        for col_name, count_sum2 in sum2.items():
            if col_name not in sum1:
                sum1[col_name] = count_sum2
                continue
            count_sum1 = sum1[col_name]
            for idx, label_sum2 in enumerate(count_sum2):
                if idx >= len(count_sum1):
                    count_sum1.append(label_sum2)
                else:
                    label_sum1 = count_sum1[idx]
                    tmp = [label_sum1[0] + label_sum2[0], label_sum1[1] + label_sum2[1]]
                    count_sum1[idx] = tmp

        return sum1
