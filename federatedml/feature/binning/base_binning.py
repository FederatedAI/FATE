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

import functools
import math
import random

from arch.api.utils import log_utils
from federatedml.feature.binning.bin_inner_param import BinInnerParam
from federatedml.feature.binning.bin_result import BinColResults, BinResults
from federatedml.feature.sparse_vector import SparseVector
from federatedml.statistic import data_overview

# from federatedml.statistic import statics

LOGGER = log_utils.getLogger()


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

    def get_data_bin(self, data_instances, split_points=None):
        """
        Apply the binning method

        Parameters
        ----------
        data_instances : DTable
            The input data

        split_points : dict.
            Each value represent for the split points for a feature. The element in each row represent for
            the corresponding split point.
            e.g.
            split_points = {'x1': [0.1, 0.2, 0.3, 0.4 ...],    # The first feature
                            'x2': [1, 2, 3, 4, ...],           # The second feature
                            ...]                         # Other features

        Returns
        -------
        data_bin_table : DTable.

            Each element represent for the corresponding bin number this feature belongs to.
            e.g. it could be:
            [{'x1': 1, 'x2': 5, 'x3': 2}
            ...
             ]
        """
        # self._init_cols(data_instances)
        is_sparse = data_overview.is_sparse_data(data_instances)

        if split_points is None:
            split_points = self.fit_split_points(data_instances)

        f = functools.partial(self.bin_data,
                              split_points=split_points,
                              cols_dict=self.bin_inner_param.bin_cols_map,
                              header=self.header,
                              is_sparse=is_sparse)
        data_bin_dict = data_instances.mapValues(f)
        return data_bin_dict

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

    def convert_feature_to_bin(self, data_instances, split_points=None):
        is_sparse = data_overview.is_sparse_data(data_instances)
        schema = data_instances.schema

        if split_points is None:
            split_points = self.bin_results.all_split_points
        else:
            for col_name, sp in split_points.items():
                self.bin_results.put_col_split_points(col_name, sp)

        if is_sparse:
            f = functools.partial(self._convert_sparse_data,
                                  bin_inner_param=self.bin_inner_param,
                                  bin_results=self.bin_results,
                                  abnormal_list=self.abnormal_list,
                                  convert_type='bin_num'
                                  )
            new_data = data_instances.mapValues(f)
        else:
            f = functools.partial(self._convert_dense_data,
                                  bin_inner_param=self.bin_inner_param,
                                  bin_results=self.bin_results,
                                  abnormal_list=self.abnormal_list,
                                  convert_type='bin_num')
            new_data = data_instances.mapValues(f)
        new_data.schema = schema
        bin_sparse = self.get_sparse_bin(self.bin_inner_param.transform_bin_indexes, split_points)
        split_points_result = self.bin_results.get_split_points_array(self.bin_inner_param.transform_bin_names)

        return new_data, split_points_result, bin_sparse

    @staticmethod
    def _convert_sparse_data(instances, bin_inner_param: BinInnerParam, bin_results: BinResults,
                             abnormal_list: list, convert_type: str = 'bin_num'):
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

    def cal_local_iv(self, data_instances, split_points=None, label_table=None):
        """
        Calculate iv attributes

        Parameters
        ----------
        data_instances : DTable
            The input data

        split_points : dict.
            Each value represent for the split points for a feature. The element in each row represent for
            the corresponding split point.
            e.g.
            split_points = {'x1': [0.1, 0.2, 0.3, 0.4 ...],    # The first feature
                            'x2': [1, 2, 3, 4, ...],           # The second feature
                            ...]                         # Other features

        label_table : DTable
            id with labels

        Returns
        -------
        Dict of IVAttributes object

        """
        # self._init_cols(data_instances)

        if split_points is None and self.split_points is None:
            split_points = self.fit_split_points(data_instances)
        elif split_points is None:
            split_points = self.split_points

        data_bin_table = self.get_data_bin(data_instances, split_points)
        if label_table is None:
            label_table = data_instances.mapValues(lambda x: x.label)
        event_count_table = label_table.mapValues(lambda x: (x, 1 - x))
        data_bin_with_label = data_bin_table.join(event_count_table, lambda x, y: (x, y))
        f = functools.partial(self.add_label_in_partition,
                              split_points=split_points,
                              cols_dict=self.bin_inner_param.bin_cols_map)

        result_sum = data_bin_with_label.mapPartitions(f)
        result_counts = result_sum.reduce(self.aggregate_partition_label)

        self.cal_iv_woe(result_counts,
                        self.params.adjustment_factor)

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
        col_bin_num = len(split_points) - 1
        for bin_num, split_point in enumerate(split_points):
            if value <= split_point:
                col_bin_num = bin_num
                break
        assert col_bin_num < len(split_points)
        return col_bin_num

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
        result_counts: DTable.
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

    @staticmethod
    def add_label_in_partition(data_bin_with_table, split_points, cols_dict):
        """
        Add all label, so that become convenient to calculate woe and iv

        Parameters
        ----------
        data_bin_with_table : DTable
            The input data, the DTable is like:
            (id, {'x1': 1, 'x2': 5, 'x3': 2}, y, 1 - y)

        split_points : dict
            Split points dict. Use to find out total bin num for each feature

        cols_dict: dict
            Record key, value pairs where key is cols' name, and value is cols' index.

        Returns
        -------
        result_sum: the result DTable. It is like:
            {'x1': [[event_count, non_event_count], [event_count, non_event_count] ... ],
             'x2': [[event_count, non_event_count], [event_count, non_event_count] ... ],
             ...
            }

        """
        result_sum = {}
        for _, datas in data_bin_with_table:
            bin_idx_dict = datas[0]
            y_combo = datas[1]

            y = y_combo[0]
            inverse_y = y_combo[1]
            for col_name, bin_idx in bin_idx_dict.items():
                result_sum.setdefault(col_name, [])
                col_sum = result_sum[col_name]
                while bin_idx >= len(col_sum):
                    col_sum.append([0, 0])
                label_sum = col_sum[bin_idx]
                label_sum[0] = label_sum[0] + y
                label_sum[1] = label_sum[1] + inverse_y
                col_sum[bin_idx] = label_sum
                result_sum[col_name] = col_sum

        return result_sum

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
                {'x1': [[event_count, non_event_count], [event_count, non_event_count] ... ],
                 'x2': [[event_count, non_event_count], [event_count, non_event_count] ... ],
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
                    tmp = (label_sum1[0] + label_sum2[0], label_sum1[1] + label_sum2[1])
                    count_sum1[idx] = tmp

        return sum1
