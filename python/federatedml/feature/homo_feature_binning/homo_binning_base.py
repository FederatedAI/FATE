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
from federatedml.framework.homo.blocks import secure_sum_aggregator
from federatedml.param.feature_binning_param import HomoFeatureBinningParam
from federatedml.statistic.data_statistics import MultivariateStatisticalSummary
from federatedml.transfer_variable.transfer_class.homo_binning_transfer_variable import HomoBinningTransferVariable


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
        rank_weight = self.aggregator.sum_model(suffix=(self.suffix, 'rank'))
        self.aggregator.send_aggregated_model(rank_weight, suffix=(self.suffix, 'rank'))


class Client(BaseBinning):
    def __init__(self, params: HomoFeatureBinningParam = None, abnormal_list=None):
        super().__init__(params, abnormal_list)
        # self.aggregator = secure_sum_aggregator.Client(enable_secure_aggregate=True)
        self.aggregator = None
        self.transfer_variable = HomoBinningTransferVariable()
        self.max_values, self.min_values = None, None
        self.suffix = None

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

    def query_values(self, summary_table, query_points):
        LOGGER.debug(f"In query_values, query_points: {query_points}")
        f = functools.partial(self._query_table,
                              query_points=query_points)
        local_ranks = dict(summary_table.map(f).collect())
        rank_arr = []
        for col_name in self.bin_inner_param.bin_names:
            LOGGER.debug(f"local_ranks: {local_ranks[col_name]}")
            rank_arr.append(local_ranks[col_name])
        rank_weight = weights.NumpyWeights(np.array(rank_arr))
        self.aggregator.send_model(rank_weight, suffix=(self.suffix, 'rank'))
        global_rank_weights = self.aggregator.get_aggregated_model(suffix=(self.suffix, 'rank')).unboxed
        res = {}
        for idx, col_name in enumerate(self.bin_inner_param.bin_names):
            if col_name == 'x0':
                LOGGER.debug(f"global_ranks: {global_rank_weights[idx]}")
            res[col_name] = global_rank_weights[idx]
        return res

    @staticmethod
    def _query_table(col_name, summary, query_points):
        queries = query_points.get(col_name)
        ranks = summary.query_value_list(queries)
        return col_name, ranks

