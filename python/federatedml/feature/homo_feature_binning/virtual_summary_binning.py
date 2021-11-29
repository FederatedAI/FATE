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

import numpy as np
import functools

from federatedml.feature.binning.quantile_tool import QuantileBinningTool
from federatedml.feature.homo_feature_binning import homo_binning_base
from federatedml.param.feature_binning_param import HomoFeatureBinningParam
# from federatedml.framework.homo.blocks import secure_sum_aggregator
from federatedml.framework.hetero.procedure import table_aggregator

from federatedml.util import LOGGER
from federatedml.util import consts


class Server(homo_binning_base.Server):
    def __init__(self, params=None, abnormal_list=None):
        super().__init__(params, abnormal_list)

    def fit_split_points(self, data=None):
        if self.aggregator is None:
            self.aggregator = table_aggregator.Server(enable_secure_aggregate=True)
        self.get_total_count()
        self.get_min_max()
        self.get_missing_count()
        self.query_values()


class Client(homo_binning_base.Client):
    def __init__(self, params: HomoFeatureBinningParam = None, abnormal_list=None, allow_duplicate=False):
        super().__init__(params, abnormal_list)
        self.allow_duplicate = allow_duplicate
        self.query_points = None
        self.global_ranks = None
        self.total_count = 0
        self.missing_count = 0

    def fit(self, data_inst):
        if self.bin_inner_param is None:
            self._setup_bin_inner_param(data_inst, self.params)
        self.total_count = self.get_total_count(data_inst)
        LOGGER.debug(f"abnormal_list: {self.abnormal_list}")

        quantile_tool = QuantileBinningTool(param_obj=self.params,
                                            abnormal_list=self.abnormal_list,
                                            allow_duplicate=self.allow_duplicate)
        quantile_tool.set_bin_inner_param(self.bin_inner_param)

        summary_table = quantile_tool.fit_summary(data_inst)

        self.get_min_max(data_inst)
        self.missing_count = self.get_missing_count(summary_table)
        self.query_points = self.init_query_points(summary_table.partitions,
                                                   split_num=self.params.sample_bins)
        self.global_ranks = self.query_values(summary_table, self.query_points)
        # self.total_count = data_inst.count()

    def fit_split_points(self, data_instances):
        if self.aggregator is None:
            self.aggregator = table_aggregator.Client(enable_secure_aggregate=True)
        self.fit(data_instances)

        query_func = functools.partial(self._query, bin_num=self.bin_num,
                                       missing_count=self.missing_count,
                                       total_count=self.total_count)
        split_point_table = self.query_points.join(self.global_ranks, lambda x, y: (x, y))
        # split_point_table = self.query_points.join(self.global_ranks, query_func)
        split_point_table = split_point_table.map(query_func)
        split_points = dict(split_point_table.collect())
        for col_name, sps in split_points.items():
            self.bin_results.put_col_split_points(col_name, sps)
        # self._query(query_ranks)
        return self.bin_results.all_split_points

    def _query(self, feature_name, values, bin_num, missing_count, total_count):
        percent_value = 1.0 / bin_num

        # calculate the split points
        percentile_rate = [i * percent_value for i in range(1, bin_num)]
        percentile_rate.append(1.0)
        this_count = total_count - missing_count[feature_name]
        query_ranks = [int(x * this_count) for x in percentile_rate]

        query_points, global_ranks = values[0], values[1]
        query_values = [x.value for x in query_points]
        query_res = []
        # query_ranks = [max(0, x - missing_count[feature_name]) for x in query_ranks]

        for rank in query_ranks:
            idx = bisect.bisect_left(global_ranks, rank)
            if idx >= len(global_ranks) - 1:
                approx_value = query_values[-1]
                query_res.append(approx_value)
            else:
                if np.fabs(query_values[idx + 1] - query_values[idx]) < consts.FLOAT_ZERO:
                    query_res.append(query_values[idx])
                elif np.fabs(global_ranks[idx + 1] - global_ranks[idx]) < consts.FLOAT_ZERO:
                    query_res.append(query_values[idx])
                else:
                    approx_value = query_values[idx] + (query_values[idx + 1] - query_values[idx]) * \
                        ((rank - global_ranks[idx]) /
                         (global_ranks[idx + 1] - global_ranks[idx]))
                    query_res.append(approx_value)
        if not self.allow_duplicate:
            query_res = sorted(set(query_res))
        return feature_name, query_res
