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

import copy
import functools
import math
import uuid
import operator

from arch.api import session
from arch.api.utils import log_utils
from federatedml.feature.binning.base_binning import BaseBinning
from federatedml.feature.binning.optimal_binning import bucket_info
from federatedml.feature.binning.optimal_binning import heap
from federatedml.feature.binning.quantile_binning import QuantileBinningTool
from federatedml.param.feature_binning_param import FeatureBinningParam, OptimalBinningParam
from federatedml.feature.binning.bin_result import BinColResults
from federatedml.statistic import data_overview
from federatedml.statistic import statics

LOGGER = log_utils.getLogger()


class OptimalBinning(BaseBinning):
    def __init__(self, params: FeatureBinningParam, abnormal_list=None):
        super().__init__(params, abnormal_list)
        self.optimal_param = params.optimal_binning_param
        self.optimal_param.adjustment_factor = params.adjustment_factor
        self.optimal_param.max_bin = params.bin_num
        self.adjustment_factor = params.adjustment_factor
        self.event_total = None
        self.non_event_total = None

    def fit_split_points(self, data_instances):
        header = data_overview.get_header(data_instances)
        self._default_setting(header)

        if (self.event_total and self.non_event_total) is None:
            self.event_total, self.non_event_total = self.get_histogram(data_instances)
        LOGGER.debug("In fit split points, event_total: {}, non_event_total: {}".format(self.event_total,
                                                                                        self.non_event_total))

        # bucket_table = self.init_bucket(data_instances)
        bucket_table = self._force_init_table(data_instances)
        sample_count = data_instances.count()
        self.fit_buckets(bucket_table, sample_count)
        self.fit_category_features(data_instances)

    def fit_buckets(self, bucket_table, sample_count):
        if self.optimal_param.metric_method in ['iv', 'gini', 'chi_square']:
            optimal_binning_method = functools.partial(self.merge_optimal_binning,
                                                       optimal_param=self.optimal_param,
                                                       sample_count=sample_count)
        else:
            optimal_binning_method = functools.partial(self.split_optimal_binning,
                                                       optimal_param=self.optimal_param,
                                                       sample_count=sample_count)
        result_bucket = bucket_table.mapValues(optimal_binning_method)
        for col_name, (bucket_list, non_mixture_num, small_size_num) in result_bucket.collect():
            split_points = [bucket.right_bound for bucket in bucket_list]
            self.bin_results.put_col_split_points(col_name, split_points)
            self.__cal_single_col_result(col_name, bucket_list)
            if self.optimal_param.mixture and non_mixture_num > 0:
                LOGGER.warning("col: {}, non_mixture_num is: {}, cannot meet mixture condition".format(
                    col_name, non_mixture_num
                ))
            if small_size_num > 0:
                LOGGER.warning("col: {}, small_size_num is: {}, cannot meet small size condition".format(
                    col_name, small_size_num
                ))
            if len(bucket_list) > self.optimal_param.max_bin:
                LOGGER.warning("col: {}, bin_num is: {}, cannot meet max-bin condition".format(
                    col_name, small_size_num
                ))
        return result_bucket

    def __cal_single_col_result(self, col_name, bucket_list):
        result_counts = [[b.event_count, b.non_event_count] for b in bucket_list]
        col_result_obj = self.woe_1d(result_counts, self.adjustment_factor)
        assert isinstance(col_result_obj, BinColResults)
        self.bin_results.put_col_results(col_name, col_result_obj)


    def init_bucket(self, data_instances):
        header = data_overview.get_header(data_instances)
        self._default_setting(header)

        quantile_param = copy.deepcopy(self.params)
        quantile_param.bin_num = self.optimal_param.init_bin_nums
        quantile_binning_tool = QuantileBinningTool(param_obj=quantile_param, allow_duplicate=False)

        every_init_bucket_nums = math.ceil(data_instances.count() / self.optimal_param.init_bin_nums)

        init_split_points = quantile_binning_tool.fit_split_points(data_instances)
        is_sparse = data_overview.is_sparse_data(data_instances)

        bucket_dict = dict()
        for col_name, sps in init_split_points.items():
            bucket_list = [bucket_info.Bucket(idx, self.adjustment_factor, right_bound=sp)
                           for idx, sp in enumerate(sps)]
            for bucket in bucket_list:
                bucket.event_total = self.event_total
                bucket.non_event_total = self.non_event_total
            bucket_list[0].set_left_neighbor(None)
            bucket_list[-1].set_right_neighbor(None)
            bucket_dict[col_name] = bucket_list

        convert_func = functools.partial(self.convert_data_to_bucket,
                                         split_points=init_split_points,
                                         headers=self.header,
                                         bucket_dict=copy.deepcopy(bucket_dict),
                                         is_sparse=is_sparse,
                                         init_bucket_nums=every_init_bucket_nums,
                                         get_bin_num_func=self.get_bin_num)
        bucket_table = data_instances.mapPartitions2(convert_func)
        bucket_table = bucket_table.reduce(self.merge_bucket_list, key_func=lambda key: key[1])
        LOGGER.debug("bucket_table: {}, length: {}".format(type(bucket_table), len(bucket_table)))
        bucket_table = [(k, v) for k, v in bucket_table.items()]
        LOGGER.debug("bucket_table: {}, length: {}".format(type(bucket_table), len(bucket_table)))
        bucket_table = session.parallelize(bucket_table, include_key=True, partition=data_instances._partitions)
        return bucket_table

    def _force_init_table(self, data_instances):
        header = data_instances.schema.get('header')
        data_dict = {}
        for h in header:
            data_dict[h] = []
        for k, instance in data_instances.collect():
            for idx, value in enumerate(instance.features):
                data_list = data_dict.get(header[idx])
                data_list.append((instance.features[idx], instance.label))
        for k, v in data_dict.items():
            v = sorted(v, key=lambda x: x[0])
            data_dict[k] = v

        LOGGER.debug("x1 values: {}".format(data_dict['x1']))

        res_bucket_list = []
        for col_name, data_list in data_dict.items():
            bucket_list = []
            last_y = data_list[0][1]
            last_boundary = -math.inf
            count = 0
            bucket_id = 0
            right_bound = data_list[0][0]
            for x, y in data_list:
                if y == last_y:
                    count += 1
                    right_bound = x
                    continue
                bucket = bucket_info.Bucket(idx=bucket_id, adjustment_factor=self.adjustment_factor, right_bound=right_bound)
                if last_y == 1:
                    bucket.event_count = count
                else:
                    bucket.non_event_count = count
                bucket.left_bound = last_boundary
                bucket.event_total = self.event_total
                bucket.non_event_total = self.non_event_total
                if bucket_id == 0:
                    bucket.set_left_neighbor(None)
                bucket_list.append(bucket)
                last_boundary = x
                right_bound = x
                count = 1
                bucket_id += 1
                last_y = y

            bucket = bucket_info.Bucket(idx=bucket_id, adjustment_factor=self.adjustment_factor, right_bound=math.inf)
            if last_y == 1:
                bucket.event_count = count
            else:
                bucket.non_event_count = count
            bucket.left_bound = last_boundary
            bucket.event_total = self.event_total
            bucket.non_event_total = self.non_event_total
            bucket.set_right_neighbor(None)
            bucket_list.append(bucket)
            res_bucket_list.append((col_name, bucket_list))

        bucket_table = session.parallelize(res_bucket_list, include_key=True, partition=data_instances._partitions)
        return bucket_table

    @staticmethod
    def get_histogram(data_instances):
        static_obj = statics.MultivariateStatisticalSummary(data_instances, cols_index=-1)
        label_historgram = static_obj.get_label_histogram()
        event_total = label_historgram.get(1, 0)
        non_event_total = label_historgram.get(0, 0)
        if event_total == 0 or non_event_total == 0:
            LOGGER.warning(f"event_total or non_event_total might have errors, event_total: {event_total},"
                           f" non_event_total: {non_event_total}")
        return event_total, non_event_total

    @staticmethod
    def assign_histogram(bucket_list, event_total, non_event_total):
        for bucket in bucket_list:
            bucket.event_total = event_total
            bucket.non_event_total = non_event_total
        return bucket_list

    @staticmethod
    def merge_bucket_list(list1, list2):
        assert len(list1) == len(list2)
        result = []
        for idx, b1 in enumerate(list1):
            b2 = list2[idx]
            result.append(b1.merge(b2))
        return result

    @staticmethod
    def convert_data_to_bucket(data_iter, split_points, headers, bucket_dict,
                               is_sparse, init_bucket_nums, get_bin_num_func):
        data_key = str(uuid.uuid1())
        for data_key, instance in data_iter:
            label = instance.label
            if not is_sparse:
                if type(instance).__name__ == 'Instance':
                    features = instance.features
                else:
                    features = instance
                data_generator = enumerate(features)
            else:
                data_generator = instance.features.get_all_data()

            for idx, col_value in data_generator:
                col_name = headers[idx]
                col_split_points = split_points[col_name]
                bin_num = get_bin_num_func(col_value, col_split_points)
                bucket = bucket_dict[col_name][bin_num]
                # while bucket.total_count >= init_bucket_nums:
                #     bin_num += 1
                #     bucket = bucket_dict[col_name][bin_num]
                bucket.add(label, col_value)
        result = []
        for col_name, bucket_list in bucket_dict.items():
            result.append(((data_key, col_name), bucket_list))
        return result

    @staticmethod
    def merge_optimal_binning(bucket_list, optimal_param: OptimalBinningParam, sample_count):

        max_item_num = math.floor(optimal_param.max_bin_pct * sample_count)
        min_item_num = math.ceil(optimal_param.min_bin_pct * sample_count)
        bucket_dict = {idx: bucket for idx, bucket in enumerate(bucket_list)}
        final_max_bin = optimal_param.max_bin

        LOGGER.debug("Get in merge optimal binning, sample_count: {}, max_item_num: {}, min_item_num: {},"
                     "final_max_bin: {}".format(sample_count, max_item_num, min_item_num, final_max_bin))
        min_heap = heap.MinHeap()

        def _add_heap_nodes(constraint=None):
            LOGGER.debug("Add heap nodes, constraint: {}, dict_length: {}".format(constraint, len(bucket_dict)))
            this_non_mixture_num = 0
            this_small_size_num = 0
            # Make bucket satisfy mixture condition

            # for i in bucket_dict.keys():
            for i in range(len(bucket_dict)):
                left_bucket = bucket_dict[i]
                right_bucket = bucket_dict.get(left_bucket.right_neighbor_idx)
                if not left_bucket.is_mixed:
                    this_non_mixture_num += 1

                if left_bucket.total_count < min_item_num:
                    this_small_size_num += 1

                if right_bucket is None:
                    continue
                # Violate maximum items constraint
                if left_bucket.total_count + right_bucket.total_count > max_item_num:
                    continue

                if constraint == 'mixture':
                    if left_bucket.is_mixed and right_bucket.is_mixed:
                        continue
                elif constraint == 'small_size':
                    if left_bucket.total_count >= min_item_num and right_bucket.total_count >= min_item_num:
                        continue
                heap_node = heap.heap_node_factory(optimal_param, left_bucket=left_bucket, right_bucket=right_bucket)
                min_heap.insert(heap_node)
            return min_heap, this_non_mixture_num, this_small_size_num

        def _update_bucket_info(b_dict):
            """
            update bucket information
            """
            order_dict = dict()
            for bucket_idx, item in b_dict.items():
                order_dict[bucket_idx] = item.left_bound

            sorted_order_dict = sorted(order_dict.items(), key=operator.itemgetter(1))

            start_idx = 0
            for item in sorted_order_dict:
                bucket_idx = item[0]
                if start_idx == bucket_idx:
                    start_idx += 1
                    continue

                b_dict[start_idx] = b_dict[bucket_idx]
                b_dict[start_idx].idx = start_idx
                start_idx += 1
                del b_dict[bucket_idx]

            bucket_num = len(b_dict)
            for i in range(bucket_num):
                if i == 0:
                    b_dict[i].set_left_neighbor(None)
                    b_dict[i].set_right_neighbor(i + 1)
                else:
                    b_dict[i].set_left_neighbor(i - 1)
                    b_dict[i].set_right_neighbor(i + 1)
            b_dict[bucket_num - 1].set_right_neighbor(None)

            return b_dict

        def _merge_heap(constraint=None, aim_var=0):
            next_id = max(bucket_dict.keys()) + 1
            while aim_var > 0 and not min_heap.is_empty:
                min_node = min_heap.pop()
                assert isinstance(min_node, heap.HeapNode)
                left_bucket = min_node.left_bucket
                right_bucket = min_node.right_bucket

                # Some buckets may be already merged
                if left_bucket.idx not in bucket_dict or right_bucket.idx not in bucket_dict:
                    continue

                new_bucket = bucket_info.Bucket(idx=next_id, adjustment_factor=optimal_param.adjustment_factor)
                new_bucket = _init_new_bucket(new_bucket, min_node)
                bucket_dict[next_id] = new_bucket
                del bucket_dict[left_bucket.idx]
                del bucket_dict[right_bucket.idx]
                aim_var = _aim_vars_decrease(constraint, new_bucket, left_bucket, right_bucket, aim_var)
                _add_node_from_new_bucket(new_bucket, constraint)
                next_id += 1
            return min_heap

        def _add_node_from_new_bucket(new_bucket: bucket_info.Bucket, constraint):
            left_bucket = bucket_dict.get(new_bucket.left_neighbor_idx)
            right_bucket = bucket_dict.get(new_bucket.right_neighbor_idx)
            if constraint == 'mixture':
                if left_bucket is not None and left_bucket.total_count + new_bucket.total_count <= max_item_num:
                    if not (left_bucket.is_mixed and new_bucket.is_mixed):
                        heap_node = heap.heap_node_factory(optimal_param, left_bucket=left_bucket,
                                                           right_bucket=new_bucket)
                        min_heap.insert(heap_node)
                if right_bucket is not None and right_bucket.total_count + new_bucket.total_count <= max_item_num:
                    if not (right_bucket.is_mixed and new_bucket.is_mixed):
                        heap_node = heap.heap_node_factory(optimal_param, left_bucket=new_bucket,
                                                           right_bucket=right_bucket)
                        min_heap.insert(heap_node)
            elif constraint == 'small_size':
                if left_bucket is not None and left_bucket.total_count + new_bucket.total_count <= max_item_num:
                    if left_bucket.total_count < min_item_num or new_bucket.total_count < min_item_num:
                        heap_node = heap.heap_node_factory(optimal_param, left_bucket=left_bucket,
                                                           right_bucket=new_bucket)
                        min_heap.insert(heap_node)
                if right_bucket is not None and right_bucket.total_count + new_bucket.total_count <= max_item_num:
                    if right_bucket.total_count < min_item_num or new_bucket.total_count < min_item_num:
                        heap_node = heap.heap_node_factory(optimal_param, left_bucket=new_bucket,
                                                           right_bucket=right_bucket)
                        min_heap.insert(heap_node)
            else:
                if left_bucket is not None and left_bucket.total_count + new_bucket.total_count <= max_item_num:
                    heap_node = heap.heap_node_factory(optimal_param, left_bucket=left_bucket,
                                                       right_bucket=new_bucket)
                    min_heap.insert(heap_node)
                if right_bucket is not None and right_bucket.total_count + new_bucket.total_count <= max_item_num:
                    heap_node = heap.heap_node_factory(optimal_param, left_bucket=new_bucket,
                                                       right_bucket=right_bucket)
                    min_heap.insert(heap_node)

        def _init_new_bucket(new_bucket: bucket_info.Bucket, min_node: heap.HeapNode):
            new_bucket.left_bound = min_node.left_bucket.left_bound
            new_bucket.right_bound = min_node.right_bucket.right_bound
            new_bucket.left_neighbor_idx = min_node.left_bucket.left_neighbor_idx
            new_bucket.right_neighbor_idx = min_node.right_bucket.right_neighbor_idx
            new_bucket.event_count = min_node.left_bucket.event_count + min_node.right_bucket.event_count
            new_bucket.non_event_count = min_node.left_bucket.non_event_count + min_node.right_bucket.non_event_count
            new_bucket.event_total = min_node.left_bucket.event_total
            new_bucket.non_event_total = min_node.left_bucket.non_event_total

            if new_bucket.left_neighbor_idx is not None:
                left_neighbor_node = bucket_dict.get(new_bucket.left_neighbor_idx)
                left_neighbor_node.right_neighbor_idx = new_bucket.idx

            if new_bucket.right_neighbor_idx is not None:
                right_neighbor_node = bucket_dict.get(new_bucket.right_neighbor_idx)
                right_neighbor_node.left_neighbor_idx = new_bucket.idx
            return new_bucket

        def _aim_vars_decrease(constraint, new_bucket: bucket_info.Bucket, left_bucket, right_bucket, aim_var):
            if constraint == 'mixture':
                if not left_bucket.is_mixed:
                    aim_var -= 1
                if not right_bucket.is_mixed:
                    aim_var -= 1
                if not new_bucket.is_mixed:
                    aim_var += 1
            elif constraint == 'small_size':
                if left_bucket.total_count < min_item_num:
                    aim_var -= 1
                if right_bucket.total_count < min_item_num:
                    aim_var -= 1
                if new_bucket.total_count < min_item_num:
                    aim_var += 1
            else:
                aim_var = len(bucket_dict) - final_max_bin
            return aim_var

        if optimal_param.mixture:
            LOGGER.debug("Before mixture add, dick length: {}".format(len(bucket_dict)))
            min_heap, non_mixture_num, small_size_num = _add_heap_nodes(constraint='mixture')
            LOGGER.debug("After mixture add, non_mixture_num: {}, min_heap size: {}".format(non_mixture_num, min_heap.size))
            min_heap = _merge_heap(constraint='mixture', aim_var=non_mixture_num)
            LOGGER.debug("After mixture merge, min_heap size: {}".format(min_heap.size))
            bucket_dict = _update_bucket_info(bucket_dict)

        LOGGER.debug("Before small_size add, dick length: {}".format(len(bucket_dict)))
        min_heap, non_mixture_num, small_size_num = _add_heap_nodes(constraint='small_size')
        LOGGER.debug("After small_size add, small_size: {}, min_heap size: {}".format(small_size_num, min_heap.size))
        min_heap = _merge_heap(constraint='small_size', aim_var=small_size_num)
        bucket_dict = _update_bucket_info(bucket_dict)

        LOGGER.debug("After small_size merge, min_heap size: {}".format(min_heap.size))
        for node_id, this_node in enumerate(min_heap.node_list):
            LOGGER.debug("node_id: {}, node_total_count: {}, left_bound: {}, right bound: {}".format(
                node_id, this_node.total_count, this_node.left_bucket.left_bound, this_node.right_bucket.right_bound
            ))

        LOGGER.debug("Before add, dick length: {}".format(len(bucket_dict)))
        min_heap, non_mixture_num, small_size_num = _add_heap_nodes()
        LOGGER.debug("After normal add, small_size: {}, min_heap size: {}".format(small_size_num, min_heap.size))
        min_heap = _merge_heap(aim_var=len(bucket_dict) - final_max_bin)
        LOGGER.debug("After normal merge, min_heap size: {}".format(min_heap.size))
        for node_id, this_node in enumerate(min_heap.node_list):
            LOGGER.debug("node_id: {}, node_total_count: {}, left_bound: {}, right bound: {}".format(
                node_id, this_node.total_count, this_node.left_bucket.left_bound, this_node.right_bucket.right_bound
            ))

        non_mixture_num = 0
        small_size_num = 0
        for i, bucket in bucket_dict.items():
            if not bucket.is_mixed:
                non_mixture_num += 1
            if bucket.total_count < min_item_num:
                small_size_num += 1
        bucket_res = list(bucket_dict.values())
        bucket_res = sorted(bucket_res, key=lambda bucket: bucket.left_bound)
        LOGGER.debug("Before return merge_optimal_binning, non_mixture_num: {}, small_size_num: {},"
                     "min_heap size: {}".format(non_mixture_num, small_size_num, min_heap.size))

        LOGGER.debug("Before return, dick length: {}".format(len(bucket_dict)))

        return bucket_res, non_mixture_num, small_size_num

    @staticmethod
    def split_optimal_binning(bucket_list, optimal_param: OptimalBinningParam, sample_count):
        min_item_num = math.ceil(optimal_param.min_bin_pct * sample_count)
        event_total = bucket_list[0].event_total
        non_event_total = bucket_list[0].non_event_total
        final_max_bin = optimal_param.max_bin

        def _compute_ks(start_idx, end_idx):
            acc_event = []
            acc_non_event = []
            event_total = 0
            non_event_total = 0
            for bucket in bucket_list[start_idx: end_idx]:
                acc_event.append(bucket.event_count + event_total)
                event_total += bucket.event_count
                acc_non_event.append(bucket.non_event_count + non_event_total)
                non_event_total += bucket.non_event_count
            acc_event_rate = [x / event_total for x in acc_event]
            acc_non_event_rate = [x / non_event_total for x in acc_non_event]
            ks_list = [math.fabs(eve - non_eve) for eve, non_eve in zip(acc_event_rate, acc_non_event_rate)]
            best_index = ks_list.index(max(ks_list))
            best_ks = ks_list[best_index]
            left_event = sum(acc_event[:best_index])
            right_event = sum(acc_event[best_index:])
            left_non_event = sum(acc_non_event[:best_index])
            right_non_event = sum(acc_non_event[best_index:])
            res_dict = {
                'left_event': left_event,
                'right_event': right_event,
                'left_non_event': left_non_event,
                'right_non_event': right_non_event,
                'left_total': left_event + left_non_event,
                'right_total': right_event + right_non_event,
                'left_is_mixed': left_event > 0 and left_non_event > 0,
                'right_is_mixed': right_event > 0 and right_non_event > 0
            }
            return best_ks, best_index, res_dict

        def _merge_buckets(start_idx, end_idx, bucket_idx):
            new_bucket = bucket_info.Bucket(idx=bucket_idx, adjustment_factor=optimal_param.adjustment_factor)
            new_bucket.event_total = event_total
            new_bucket.non_event_total = non_event_total
            for bucket in bucket_list[start_idx: end_idx]:
                new_bucket = new_bucket.merge(bucket)
            return new_bucket

        res_split_index = []
        to_split_pair = [(0, len(bucket_list))]

        # iteratively split
        while len(to_split_pair) > 0:
            if len(res_split_index) >= final_max_bin - 1:
                break
            start, end = to_split_pair.pop(0)
            if start >= end:
                continue
            best_ks, best_index, res_dict = _compute_ks(start, end)
            if optimal_param.mixture:
                if not (res_dict.get('left_is_mixed') and res_dict.get('right_is_mixed')):
                    continue
            if res_dict.get('left_total') < min_item_num or res_dict.get('right_total') < min_item_num:
                continue
            res_split_index.append(best_index)
            if res_dict.get('right_total') > res_dict.get('left_total'):
                to_split_pair.append((best_index + 1, end))
                to_split_pair.append((start, best_index))
            else:
                to_split_pair.append((start, best_index))
                to_split_pair.append((best_index + 1, end))

        if len(res_split_index) == 0:
            LOGGER.warning("Best ks optimal binning fail to split. Take middle split point instead")
            res_split_index.append(len(bucket_list) // 2)
        res_split_index = sorted(res_split_index)
        res_split_index.append(len(bucket_list))
        start = 0
        bucket_res = []
        non_mixture_num = 0
        small_size_num = 0
        for bucket_idx, end in enumerate(res_split_index):
            new_bucket = _merge_buckets(start, end, bucket_idx)
            bucket_res.append(new_bucket)
            if not new_bucket.is_mixed:
                non_mixture_num += 1
            if new_bucket.total_count < min_item_num:
                small_size_num += 1
            start = end
        return bucket_res, non_mixture_num, small_size_num

    def bin_sum_to_bucket_list(self, bin_sum, partitions):
        """
        Convert bin sum result, which typically get from host, to bucket list
        Parameters
        ----------
        bin_sum : dict
           {'x1': [[event_count, non_event_count], [event_count, non_event_count] ... ],
             'x2': [[event_count, non_event_count], [event_count, non_event_count] ... ],
             ...
            }

        partitions: int
            Indicate partitions for created table.

        Returns
        -------
        A DTable whose keys are feature names and values are bucket lists
        """
        bucket_dict = dict()
        for col_name, bin_res_list in bin_sum.items():
            bucket_list = [bucket_info.Bucket(idx, self.adjustment_factor) for idx in range(len(bin_res_list))]
            bucket_list[0].set_left_neighbor(None)
            bucket_list[-1].set_right_neighbor(None)
            for b_idx, bucket in enumerate(bucket_list):
                bucket.event_count = bin_res_list[b_idx][0]
                bucket.non_event_count = bin_res_list[b_idx][1]
                bucket.left_bound = b_idx - 1
                bucket.right_bound = b_idx
                bucket.event_total = self.event_total
                bucket.non_event_total = self.non_event_total
            bucket_dict[col_name] = bucket_list

        result = []
        for col_name, bucket_list in bucket_dict.items():
            result.append((col_name, bucket_list))
        result_table = session.parallelize(result,
                                           include_key=True,
                                           partition=partitions)
        return result_table
