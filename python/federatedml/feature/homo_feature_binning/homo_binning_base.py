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

import numpy as np
from federatedml.util import LOGGER

from federatedml.feature.binning.base_binning import BaseBinning
from federatedml.framework import weights
from fate_arch.session import computing_session as session
from federatedml.param.feature_binning_param import HomoFeatureBinningParam
from federatedml.statistic.data_statistics import MultivariateStatisticalSummary
from federatedml.transfer_variable.transfer_class.homo_binning_transfer_variable import HomoBinningTransferVariable
from federatedml.util import consts


class SplitPointNode(object):
    def __init__(self, value, min_value, max_value, aim_rank=None, allow_error_rank=0, last_rank=-1):
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.aim_rank = aim_rank
        self.allow_error_rank = allow_error_rank
        self.last_rank = last_rank
        self.fixed = False

    def set_aim_rank(self, rank):
        self.aim_rank = rank

    def create_right_new(self):
        value = (self.value + self.max_value) / 2
        if np.fabs(value - self.value) <= consts.FLOAT_ZERO * 0.9:
            self.value += consts.FLOAT_ZERO * 0.9
            self.fixed = True
            return self
        min_value = self.value
        return SplitPointNode(value, min_value, self.max_value, self.aim_rank, self.allow_error_rank)

    def create_left_new(self):
        value = (self.value + self.min_value) / 2
        if np.fabs(value - self.value) <= consts.FLOAT_ZERO * 0.9:
            self.value += consts.FLOAT_ZERO * 0.9
            self.fixed = True
            return self
        max_value = self.value
        return SplitPointNode(value, self.min_value, max_value, self.aim_rank, self.allow_error_rank)


class RankArray(object):
    def __init__(self, rank_array, error_rank, last_rank_array=None):
        self.rank_array = rank_array
        self.last_rank_array = last_rank_array
        self.error_rank = error_rank
        self.all_fix = False
        self.fixed_array = np.zeros(len(self.rank_array), dtype=bool)
        self._compare()

    def _compare(self):
        if self.last_rank_array is None:
            return
        else:
            self.fixed_array = abs(self.rank_array - self.last_rank_array) < self.error_rank
            assert isinstance(self.fixed_array, np.ndarray)
            if (self.fixed_array).all():
                self.all_fix = True

    def __iadd__(self, other: 'RankArray'):
        for idx, is_fixed in enumerate(self.fixed_array):
            if not is_fixed:
                self.rank_array[idx] += other.rank_array[idx]
        self._compare()
        return self

    def __add__(self, other: 'RankArray'):
        res_array = []
        for idx, is_fixed in enumerate(self.fixed_array):
            if not is_fixed:
                res_array.append(self.rank_array[idx] + other.rank_array[idx])
            else:
                res_array.append(self.rank_array[idx])
        return RankArray(np.array(res_array), self.error_rank, self.last_rank_array)


class Server(BaseBinning):
    def __init__(self, params=None, abnormal_list=None):
        super().__init__(params, abnormal_list)
        # self.aggregator = secure_sum_aggregator.Server(enable_secure_aggregate=True)
        self.aggregator = None

        self.transfer_variable = HomoBinningTransferVariable()
        self.suffix = None

    def set_suffix(self, suffix):
        self.suffix = suffix

    def set_transfer_variable(self, variable):
        self.transfer_variable = variable

    def set_aggregator(self, aggregator):
        self.aggregator = aggregator

    def get_total_count(self):
        total_count = self.aggregator.sum_model(suffix=(self.suffix, 'total_count'))
        self.aggregator.send_aggregated_model(total_count, suffix=(self.suffix, 'total_count'))
        return total_count

    def get_missing_count(self):
        missing_count = self.aggregator.sum_model(suffix=(self.suffix, 'missing_count'))
        self.aggregator.send_aggregated_model(missing_count, suffix=(self.suffix, 'missing_count'))
        return missing_count

    def get_min_max(self):
        local_values = self.transfer_variable.local_static_values.get(suffix=(self.suffix, "min-max"))
        max_array, min_array = [], []
        for local_max, local_min in local_values:
            max_array.append(local_max)
            min_array.append(local_min)
        max_values = np.max(max_array, axis=0)
        min_values = np.min(min_array, axis=0)
        self.transfer_variable.global_static_values.remote((max_values, min_values),
                                                           suffix=(self.suffix, "min-max"))
        return min_values, max_values

    def query_values(self):
        rank_weight = self.aggregator.aggregate_tables(suffix=(self.suffix, 'rank'))
        self.aggregator.send_aggregated_tables(rank_weight, suffix=(self.suffix, 'rank'))


class Client(BaseBinning):
    def __init__(self, params: HomoFeatureBinningParam = None, abnormal_list=None):
        super().__init__(params, abnormal_list)
        # self.aggregator = secure_sum_aggregator.Client(enable_secure_aggregate=True)
        self.aggregator = None
        self.transfer_variable = HomoBinningTransferVariable()
        self.max_values, self.min_values = None, None
        self.suffix = None
        self.total_count = 0

    def set_suffix(self, suffix):
        self.suffix = suffix

    def set_transfer_variable(self, variable):
        self.transfer_variable = variable

    def set_aggregator(self, aggregator):
        self.aggregator = aggregator

    def get_total_count(self, data_inst):
        count = data_inst.count()
        count_weight = weights.NumericWeights(count)
        self.aggregator.send_model(count_weight, suffix=(self.suffix, 'total_count'))
        total_count = self.aggregator.get_aggregated_model(suffix=(self.suffix, 'total_count')).unboxed
        return total_count

    def get_missing_count(self, summary_table):
        missing_table = summary_table.mapValues(lambda x: x.missing_count)
        missing_value_counts = dict(missing_table.collect())
        missing_weight = weights.DictWeights(missing_value_counts)
        self.aggregator.send_model(missing_weight, suffix=(self.suffix, 'missing_count'))
        missing_counts = self.aggregator.get_aggregated_model(suffix=(self.suffix, 'missing_count')).unboxed
        return missing_counts

    def get_min_max(self, data_inst):
        """
        Get max and min value of each selected columns

        Returns:
            max_values, min_values: dict
            eg. {"x1": 10, "x2": 3, ... }

        """
        if self.max_values and self.min_values:
            return self.max_values, self.min_values
        statistic_obj = MultivariateStatisticalSummary(data_inst,
                                                       cols_index=self.bin_inner_param.bin_indexes,
                                                       abnormal_list=self.abnormal_list,
                                                       error=self.params.error)
        max_values = statistic_obj.get_max()
        min_values = statistic_obj.get_min()
        max_list = [max_values[x] for x in self.bin_inner_param.bin_names]
        min_list = [min_values[x] for x in self.bin_inner_param.bin_names]
        local_min_max_values = (max_list, min_list)
        self.transfer_variable.local_static_values.remote(local_min_max_values,
                                                          suffix=(self.suffix, "min-max"))
        self.max_values, self.min_values = self.transfer_variable.global_static_values.get(
            idx=0, suffix=(self.suffix, "min-max"))
        return self.max_values, self.min_values

    def init_query_points(self, partitions, split_num, error_rank=1, need_first=True):
        query_points = []
        for idx, col_name in enumerate(self.bin_inner_param.bin_names):
            max_value = self.max_values[idx]
            min_value = self.min_values[idx]
            sps = np.linspace(min_value, max_value, split_num)

            if not need_first:
                sps = sps[1:]

            split_point_array = [SplitPointNode(sps[i], min_value, max_value, allow_error_rank=error_rank)
                                 for i in range(len(sps))]
            query_points.append((col_name, split_point_array))
        query_points_table = session.parallelize(query_points,
                                                 include_key=True,
                                                 partition=partitions)
        return query_points_table

    def query_values(self, summary_table, query_points):
        local_ranks = summary_table.join(query_points, self._query_table)

        self.aggregator.send_table(local_ranks, suffix=(self.suffix, 'rank'))
        global_rank = self.aggregator.get_aggregated_table(suffix=(self.suffix, 'rank'))
        global_rank = global_rank.mapValues(lambda x: np.array(x, dtype=int))
        return global_rank

    @staticmethod
    def _query_table(summary, query_points):
        queries = [x.value for x in query_points]
        original_idx = np.argsort(np.argsort(queries))
        queries = np.sort(queries)
        ranks = summary.query_value_list(queries)
        ranks = np.array(ranks)[original_idx]
        return np.array(ranks, dtype=int)
