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
import operator

import numpy as np

from federatedml.feature.binning.base_binning import BaseBinning
from federatedml.feature.binning.bin_result import BinColResults, MultiClassBinResult
from federatedml.statistic import data_overview
from federatedml.feature.sparse_vector import SparseVector
from federatedml.cipher_compressor.compressor import PackingCipherTensor
from federatedml.util import LOGGER


class IvCalculator(object):
    def __init__(self, adjustment_factor, role, party_id):
        self.adjustment_factor = adjustment_factor
        self.role = role
        self.party_id = party_id

    def cal_local_iv(self, data_instances, split_points,
                     labels=None, label_counts=None, bin_cols_map=None,
                     label_table=None):
        """
        data_bin_table : Table.

            Each element represent for the corresponding bin number this feature belongs to.
            e.g. it could be:
            [{'x1': 1, 'x2': 5, 'x3': 2}
            ...
             ]
        Returns:
            MultiClassBinResult object
        """
        header = data_instances.schema.get("header")
        if bin_cols_map is None:
            bin_cols_map = {name: idx for idx, name in enumerate(header)}
            bin_indexes = [idx for idx, _ in enumerate(header)]
        else:
            bin_indexes = []
            for h in header:
                if h in bin_cols_map:
                    bin_indexes.append(bin_cols_map[h])
        if label_counts is None:
            label_counts = data_overview.get_label_count(data_instances)
            labels = list(label_counts.keys())
            label_counts = [label_counts[k] for k in labels]

        data_bin_table = BaseBinning.get_data_bin(data_instances, split_points, bin_cols_map)
        sparse_bin_points = BaseBinning.get_sparse_bin(bin_indexes, split_points, header)
        sparse_bin_points = {header[k]: v for k, v in sparse_bin_points.items()}

        if label_table is None:
            label_table = self.convert_label(data_instances, labels)

        result_counts = self.cal_bin_label(data_bin_table, sparse_bin_points, label_table, label_counts)
        multi_bin_res = self.cal_iv_from_counts(result_counts, labels,
                                                role=self.role,
                                                party_id=self.party_id)
        for col_name, sp in split_points.items():
            multi_bin_res.put_col_split_points(col_name, sp)
        return multi_bin_res

    def cal_iv_from_counts(self, result_counts, labels, role, party_id):
        result = MultiClassBinResult(labels)
        result.set_role_party(role, party_id)
        if len(labels) == 2:
            col_result_obj_dict = self.cal_single_label_iv_woe(result_counts,
                                                               self.adjustment_factor)
            for col_name, bin_col_result in col_result_obj_dict.items():
                result.put_col_results(col_name=col_name, col_results=bin_col_result)
        else:
            for label_idx, y in enumerate(labels):
                this_result_counts = self.mask_label(result_counts, label_idx)
                col_result_obj_dict = self.cal_single_label_iv_woe(this_result_counts,
                                                                   self.adjustment_factor)
                for col_name, bin_col_result in col_result_obj_dict.items():
                    result.put_col_results(col_name=col_name, col_results=bin_col_result, label_idx=label_idx)
        return result

    @staticmethod
    def mask_label(result_counts, label_idx):
        def _mask(counts):
            res = []
            for c in counts:
                res.append(np.array([c[label_idx], np.sum(c) - c[label_idx]]))
            return res

        return result_counts.mapValues(_mask)

    def cal_bin_label(self, data_bin_table, sparse_bin_points, label_table, label_counts):
        """

        data_bin_table : Table.
            Each element represent for the corresponding bin number this feature belongs to.
            e.g. it could be:
            [{'x1': 1, 'x2': 5, 'x3': 2}
            ...
             ]

        sparse_bin_points: dict
            Dict of sparse bin num
                {"x0": 2, "x1": 3, "x2": 5 ... }

        label_table : Table
            id with labels

        Returns:
            Table with value:
            [[label_0_sum, label_1_sum, ...], [label_0_sum, label_1_sum, ...] ... ]
        """
        data_bin_with_label = data_bin_table.join(label_table, lambda x, y: (x, y))
        f = functools.partial(self.add_label_in_partition,
                              sparse_bin_points=sparse_bin_points)

        result_counts = data_bin_with_label.mapReducePartitions(f, self.aggregate_partition_label)

        return result_counts

    def cal_single_label_iv_woe(self, result_counts, adjustment_factor):
        """
        Given event count information calculate iv information

        Parameters
        ----------
        result_counts: dict or table.
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
        if isinstance(result_counts, dict):
            col_result_obj_dict = {}
            for col_name, data_event_count in result_counts.items():
                col_result_obj = self.woe_1d(data_event_count, adjustment_factor)
                col_result_obj_dict[col_name] = col_result_obj
        else:
            woe_1d = functools.partial(self.woe_1d, adjustment_factor=adjustment_factor)
            col_result_obj_dict = dict(result_counts.mapValues(woe_1d).collect())
        return col_result_obj_dict

    @staticmethod
    def fill_sparse_result(col_name, static_nums, sparse_bin_points, label_counts):
        """
        Parameters
        ----------
        col_name: str
            current col_name, use to obtain sparse point

        static_nums :  list.
            It is like:
                [[label_0_sum, label_1_sum, ...], [label_0_sum, label_1_sum, ...] ... ]
                where the bin of sparse point located in is empty.

        sparse_bin_points : dict
            Dict of sparse bin num
                {"x1": 2, "x2": 3, "x3": 5 ... }

        label_counts: np.array
            eg. [100, 200, ...]

        Returns
        -------
        The format is same as static_nums.
        """

        curt_all = functools.reduce(lambda x, y: x + y, static_nums)
        sparse_bin = sparse_bin_points.get(col_name)
        static_nums[sparse_bin] = label_counts - curt_all
        return col_name, static_nums

    @staticmethod
    def combine_labels(result_counts, idx):
        """
        result_counts: Table
            [[label_0_sum, label_1_sum, ...], [label_0_sum, label_1_sum, ...] ... ]

        idx: int

        Returns:

        """

    @staticmethod
    def add_label_in_partition(data_bin_with_table, sparse_bin_points):
        """
        Add all label, so that become convenient to calculate woe and iv

        Parameters
        ----------
        data_bin_with_table : Table
            The input data, the Table is like:
            (id, {'x1': 1, 'x2': 5, 'x3': 2}, y)
            where y = [is_label_0, is_label_1, ...]  which is one-hot format array of label

        sparse_bin_points: dict
            Dict of sparse bin num
                {0: 2, 1: 3, 2:5 ... }

        Returns
        -------
            ['x1', [[label_0_sum, label_1_sum, ...], [label_0_sum, label_1_sum, ...] ... ],
             'x2', [[label_0_sum, label_1_sum, ...], [label_0_sum, label_1_sum, ...] ... ],
             ...
            ]

        """
        result_sum = {}
        for _, datas in data_bin_with_table:
            bin_idx_dict = datas[0]
            y = datas[1]
            for col_name, bin_idx in bin_idx_dict.items():
                result_sum.setdefault(col_name, [])
                col_sum = result_sum[col_name]
                while bin_idx >= len(col_sum):
                    if isinstance(y, PackingCipherTensor):
                        zero_y = np.zeros(y.dim)
                        col_sum.append(PackingCipherTensor(zero_y.tolist()))
                    else:
                        col_sum.append(np.zeros(len(y)))

                # if bin_idx == sparse_bin_points[col_name]:
                #     continue
                col_sum[bin_idx] = col_sum[bin_idx] + y
        return list(result_sum.items())

    @staticmethod
    def aggregate_partition_label(sum1, sum2):
        """
        Used in reduce function. Aggregate the result calculate from each partition.

        Parameters
        ----------
        sum1 :  list.
            It is like:
            [[label_0_sum, label_1_sum, ...], [label_0_sum, label_1_sum, ...] ... ]
        sum2 : list
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

        for idx, label_sum2 in enumerate(sum2):
            if idx >= len(sum1):
                sum1.append(label_sum2)
            else:
                sum1[idx] = sum1[idx] + label_sum2
        return sum1

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
        for bin_res in data_event_count:
            if len(bin_res) != 2:
                raise ValueError(f"bin_res should has length of 2,"
                                 f" data_event_count: {data_event_count}, bin_res: {bin_res}")
            event_total += bin_res[0]
            non_event_total += bin_res[1]

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

            event_count_array.append(int(event_count))
            non_event_count_array.append(int(non_event_count))
            event_rate_array.append(event_rate)
            non_event_rate_array.append(non_event_rate)
            woe_array.append(woe_i)
            iv_i = (event_rate - non_event_rate) * woe_i
            iv_array.append(iv_i)
            iv += iv_i
        return BinColResults(woe_array=woe_array, iv_array=iv_array, event_count_array=event_count_array,
                             non_event_count_array=non_event_count_array,
                             event_rate_array=event_rate_array, non_event_rate_array=non_event_rate_array, iv=iv)

    @staticmethod
    def statistic_label(data_instances):
        label_counts = data_overview.get_label_count(data_instances)
        label_elements = list(label_counts.keys())
        label_counts = [label_counts[k] for k in label_elements]
        return label_elements, label_counts

    @staticmethod
    def convert_label(data_instances, label_elements):
        def _convert(instance):
            res_labels = np.zeros(len(label_elements))
            res_labels[label_elements.index(instance.label)] = 1
            return res_labels

        label_table = data_instances.mapValues(_convert)
        return label_table

    @staticmethod
    def woe_transformer(data_instances, bin_inner_param, multi_class_bin_res: MultiClassBinResult,
                        abnormal_list=None):
        if abnormal_list is None:
            abnormal_list = []
        bin_res = multi_class_bin_res.bin_results[0]
        transform_cols_idx = bin_inner_param.transform_bin_indexes
        split_points_dict = bin_res.all_split_points
        is_sparse = data_overview.is_sparse_data(data_instances)

        def convert(instances):
            if is_sparse:
                all_data = instances.features.get_all_data()
                indice = []
                sparse_value = []
                data_shape = instances.features.get_shape()
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
                        col_results = bin_res.all_cols_results.get(col_name)
                        woe_value = col_results.woe_array[bin_num]
                        sparse_value.append(woe_value)
                    else:
                        indice.append(col_idx)
                        sparse_value.append(col_value)
                sparse_vector = SparseVector(indice, sparse_value, data_shape)
                instances.features = sparse_vector
            else:
                features = instances.features
                assert isinstance(features, np.ndarray)
                transform_cols_idx_set = set(transform_cols_idx)

                for col_idx, col_value in enumerate(features):
                    if col_idx in transform_cols_idx_set:
                        if col_value in abnormal_list:
                            features[col_idx] = col_value
                            continue
                        col_name = bin_inner_param.header[col_idx]
                        split_points = split_points_dict[col_name]
                        bin_num = BaseBinning.get_bin_num(col_value, split_points)
                        col_results = bin_res.all_cols_results.get(col_name)
                        woe_value = col_results.woe_array[bin_num]
                        features[col_idx] = woe_value
                instances.features = features
            return instances

        return data_instances.mapValues(convert)

    @staticmethod
    def check_containing_missing_value(data_instances):
        is_sparse = data_overview.is_sparse_data(data_instances)

        def _sparse_check(instance):
            result = set()
            sparse_data = instance.features.get_all_data()
            for col_idx, col_value in sparse_data:
                if np.isnan(col_value):
                    result.add(col_idx)
            return result

        if is_sparse:
            has_missing_value = data_instances.mapValues(_sparse_check).reduce(
                lambda a, b: a.union(b)
            )
        else:
            has_missing_value = data_instances.mapValues(lambda x: x.features).reduce(operator.add)
            has_missing_value = {idx for idx, value in enumerate(has_missing_value) if np.isnan(value)}
        return has_missing_value
