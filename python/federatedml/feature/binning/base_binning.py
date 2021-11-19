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
import functools
import math
import random
import copy
import numpy as np

from federatedml.feature.binning.bin_inner_param import BinInnerParam
from federatedml.feature.binning.bin_result import BinColResults, SplitPointsResult
from federatedml.statistic.data_overview import get_header
from federatedml.feature.sparse_vector import SparseVector
from federatedml.statistic import data_overview
from federatedml.util import LOGGER
from federatedml.feature.fate_element_type import NoneType

# from federatedml.statistic import statics


class BaseBinning(object):
    """
    This is use for discrete data so that can transform data or use information for feature selection.
    """

    def __init__(self, params=None, abnormal_list=None, labels=None):
        self.bin_inner_param: BinInnerParam = None
        self.is_multi_class = False
        self.bin_results = SplitPointsResult()
        if params is None:
            return
        self.params = params
        self.bin_num = params.bin_num
        if abnormal_list is None:
            self.abnormal_list = []
        else:
            self.abnormal_list = abnormal_list
        self.split_points = None

    @property
    def header(self):
        return self.bin_inner_param.header

    # @property
    # def split_points(self):
    #     return self.bin_results.all_split_points

    def _default_setting(self, header):
        if self.bin_inner_param is not None:
            return
        self.bin_inner_param = BinInnerParam()

        self.bin_inner_param.set_header(header)
        if self.params.bin_indexes == -1:
            self.bin_inner_param.set_bin_all()
        else:
            self.bin_inner_param.add_bin_indexes(self.params.bin_indexes)
            self.bin_inner_param.add_bin_names(self.params.bin_names)

        self.bin_inner_param.add_category_indexes(self.params.category_indexes)
        self.bin_inner_param.add_category_names(self.params.category_names)

        if self.params.transform_param.transform_cols == -1:
            self.bin_inner_param.set_transform_all()
        else:
            self.bin_inner_param.add_transform_bin_indexes(self.params.transform_param.transform_cols)
            self.bin_inner_param.add_transform_bin_names(self.params.transform_param.transform_names)

    def fit_split_points(self, data_instances):
        """
        Get split points

        Parameters
        ----------
        data_instances : Table
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

    @staticmethod
    def get_data_bin(data_instances, split_points, bin_cols_map):
        """
        Apply the binning method

        Parameters
        ----------
        data_instances : Table
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
        data_bin_table : Table.

            Each element represent for the corresponding bin number this feature belongs to.
            e.g. it could be:
            [{'x1': 1, 'x2': 5, 'x3': 2}
            ...
             ]
        """
        # self._init_cols(data_instances)
        is_sparse = data_overview.is_sparse_data(data_instances)
        header = data_instances.schema.get('header')

        f = functools.partial(BaseBinning.bin_data,
                              split_points=split_points,
                              cols_dict=bin_cols_map,
                              header=header,
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
        header = get_header(data_instances)
        bin_sparse = self.get_sparse_bin(self.bin_inner_param.transform_bin_indexes, split_points, header)
        split_points_result = self.bin_results.get_split_points_array(self.bin_inner_param.transform_bin_names)

        return new_data, split_points_result, bin_sparse

    def _setup_bin_inner_param(self, data_instances, params):
        if self.bin_inner_param is not None:
            return
        self.bin_inner_param = BinInnerParam()

        header = get_header(data_instances)
        LOGGER.debug("_setup_bin_inner_param, get header length: {}".format(len(self.header)))

        self.schema = data_instances.schema
        self.bin_inner_param.set_header(header)
        if params.bin_indexes == -1:
            self.bin_inner_param.set_bin_all()
        else:
            self.bin_inner_param.add_bin_indexes(params.bin_indexes)
            self.bin_inner_param.add_bin_names(params.bin_names)

        self.bin_inner_param.add_category_indexes(params.category_indexes)
        self.bin_inner_param.add_category_names(params.category_names)

        if params.transform_param.transform_cols == -1:
            self.bin_inner_param.set_transform_all()
        else:
            self.bin_inner_param.add_transform_bin_indexes(params.transform_param.transform_cols)
            self.bin_inner_param.add_transform_bin_names(params.transform_param.transform_names)
        self.set_bin_inner_param(self.bin_inner_param)

    @staticmethod
    def _convert_sparse_data(instances, bin_inner_param: BinInnerParam, bin_results: SplitPointsResult,
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
                else:
                    sparse_value.append(col_value)
            else:
                indice.append(col_idx)
                sparse_value.append(col_value)

        sparse_vector = SparseVector(indice, sparse_value, data_shape)
        instances.features = sparse_vector
        return instances

    @staticmethod
    def get_sparse_bin(transform_cols_idx, split_points_dict, header):
        """
        Get which bins the 0 located at for each column.

        Returns
        -------
        Dict of sparse bin num
            {0: 2, 1: 3, 2:5 ... }
        """
        result = {}
        for col_idx in transform_cols_idx:
            col_name = header[col_idx]
            split_points = split_points_dict[col_name]
            sparse_bin_num = BaseBinning.get_bin_num(0, split_points)
            result[col_idx] = sparse_bin_num
        return result

    @staticmethod
    def _convert_dense_data(instances, bin_inner_param: BinInnerParam, bin_results: SplitPointsResult,
                            abnormal_list: list, convert_type: str = 'bin_num'):
        instances = copy.deepcopy(instances)
        features = instances.features
        transform_cols_idx = bin_inner_param.transform_bin_indexes
        split_points_dict = bin_results.all_split_points

        transform_cols_idx_set = set(transform_cols_idx)

        for col_idx, col_value in enumerate(features):
            if col_idx in transform_cols_idx_set:
                if col_value in abnormal_list:
                    features[col_idx] = col_value
                    continue
                col_name = bin_inner_param.header[col_idx]
                split_points = split_points_dict[col_name]
                bin_num = BaseBinning.get_bin_num(col_value, split_points)
                if convert_type == 'bin_num':
                    features[col_idx] = bin_num
                else:
                    features[col_idx] = col_value

        instances.features = features
        return instances

    @staticmethod
    def convert_bin_counts_table(result_counts, idx):
        """
        Given event count information calculate iv information

        Parameters
        ----------
        result_counts: table.
            It is like:
                ('x1': [[label_0_count, label_1_count, ...], [label_0_count, label_1_count, ...] ... ],
                 'x2': [[label_0_count, label_1_count, ...], [label_0_count, label_1_count, ...] ... ],
                 ...
                )

        idx: int

        Returns
        -------
        ('x1': [[event_count, non_event_count], [event_count, non_event_count] ... ],
         'x2': [[event_count, non_event_count], [event_count, non_event_count] ... ],
         ...
        )
        """

        def _convert(list_counts):
            res = []
            for c_array in list_counts:
                event_count = c_array[idx]
                non_event_count = np.sum(c_array) - event_count
                res.append([event_count, non_event_count])
            return res

        return result_counts.mapValues(_convert)

    @staticmethod
    def fill_sparse_result(col_name, static_nums, sparse_bin_points, label_counts):
        """
        Parameters
        ----------
        static_nums :  list.
            It is like:
                [[event_count, total_num], [event_count, total_num] ... ]


        sparse_bin_points : dict
            Dict of sparse bin num
                {"x1": 2, "x2": 3, "x3": 5 ... }

        label_counts: np.array
            eg. [100, 200, ...]

        Returns
        -------
        The format is same as result_counts.
        """

        curt_all = functools.reduce(lambda x, y: x + y, static_nums)
        sparse_bin = sparse_bin_points.get(col_name)
        static_nums[sparse_bin] = label_counts - curt_all
        return col_name, static_nums

    @staticmethod
    def bin_data(instance, split_points, cols_dict, header, is_sparse):
        """
        Apply the binning method

        Parameters
        ----------
        instance : Table
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

        header: list
            header of Table

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
        if np.isnan(value):
            return len(split_points)
        sp = split_points[:-1]
        col_bin_num = bisect.bisect_left(sp, value)
        # col_bin_num = bisect.bisect_left(split_points, value)
        return col_bin_num

    @staticmethod
    def add_label_in_partition_bak(data_bin_with_table, sparse_bin_points):
        """
        Add all label, so that become convenient to calculate woe and iv

        Parameters
        ----------
        data_bin_with_table : Table
            The input data, the Table is like:
            (id, {'x1': 1, 'x2': 5, 'x3': 2}, y)

        sparse_bin_points: dict
            Dict of sparse bin num
                {0: 2, 1: 3, 2:5 ... }

        Returns
        -------
        result_sum: the result Table. It is like:
            {'x1': [[event_count, total_num], [event_count, total_num] ... ],
             'x2': [[event_count, total_num], [event_count, total_num] ... ],
             ...
            }

        """
        result_sum = {}
        for _, datas in data_bin_with_table:
            bin_idx_dict = datas[0]
            y = datas[1]

            # y = y_combo[0]
            # inverse_y = y_combo[1]
            for col_name, bin_idx in bin_idx_dict.items():
                result_sum.setdefault(col_name, [])
                col_sum = result_sum[col_name]
                while bin_idx >= len(col_sum):
                    col_sum.append([0, 0])
                if bin_idx == sparse_bin_points[col_name]:
                    continue
                label_sum = col_sum[bin_idx]
                label_sum[0] = label_sum[0] + y
                label_sum[1] = label_sum[1] + 1
                col_sum[bin_idx] = label_sum
                result_sum[col_name] = col_sum

        return list(result_sum.items())
