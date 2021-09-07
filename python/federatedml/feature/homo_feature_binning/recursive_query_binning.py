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

import numpy as np

from fate_arch.session import computing_session as session
from federatedml.feature.binning.quantile_tool import QuantileBinningTool
from federatedml.feature.homo_feature_binning import homo_binning_base
from federatedml.framework.hetero.procedure import table_aggregator
from federatedml.param.feature_binning_param import HomoFeatureBinningParam
from federatedml.util import consts
from federatedml.util import LOGGER
import copy
import operator
import functools


class Server(homo_binning_base.Server):
    def __init__(self, params: HomoFeatureBinningParam, abnormal_list=None):
        super().__init__(params, abnormal_list)

    def fit_split_points(self, data=None):
        if self.aggregator is None:
            self.aggregator = table_aggregator.Server(enable_secure_aggregate=False)
        self.get_total_count()
        self.get_min_max()
        self.get_missing_count()
        self.set_suffix(-1)
        self.query_values()
        n_iter = 0
        while n_iter < self.params.max_iter:
            self.set_suffix(n_iter)
            is_converge = self.transfer_variable.is_converge.get(suffix=self.suffix)[0]
            if is_converge:
                break
            self.query_values()
            n_iter += 1


class Client(homo_binning_base.Client):
    def __init__(self, role, params: HomoFeatureBinningParam = None,
                 abnormal_list=None, allow_duplicate=False):
        super().__init__(params, abnormal_list)
        self.allow_duplicate = allow_duplicate
        self.global_ranks = {}
        self.total_count = 0
        self.missing_counts = 0
        self.error = params.error
        self.error_rank = None
        self.role = role

    def fit_split_points(self, data_instances):
        if self.aggregator is None:
            self.aggregator = table_aggregator.Client(enable_secure_aggregate=False)
        if self.bin_inner_param is None:
            self._setup_bin_inner_param(data_instances, self.params)
        self.total_count = self.get_total_count(data_instances)
        self.error_rank = np.ceil(self.error * self.total_count)
        LOGGER.debug(f"abnormal_list: {self.abnormal_list}")
        quantile_tool = QuantileBinningTool(param_obj=self.params,
                                            abnormal_list=self.abnormal_list,
                                            allow_duplicate=self.allow_duplicate)
        quantile_tool.set_bin_inner_param(self.bin_inner_param)
        summary_table = quantile_tool.fit_summary(data_instances)

        self.get_min_max(data_instances)
        self.missing_counts = self.get_missing_count(summary_table)
        split_points_table = self._recursive_querying(summary_table)
        split_points = dict(split_points_table.collect())
        for col_name, sps in split_points.items():
            sp = [x.value for x in sps]
            if not self.allow_duplicate:
                sp = sorted(set(sp))
                res = [sp[0] if np.fabs(sp[0]) > consts.FLOAT_ZERO else 0.0]
                last = sp[0]
                for v in sp[1:]:
                    if np.fabs(v) < consts.FLOAT_ZERO:
                        v = 0.0
                    if np.abs(v - last) > consts.FLOAT_ZERO:
                        res.append(v)
                    last = v
                sp = np.array(res)
            self.bin_results.put_col_split_points(col_name, sp)
        return self.bin_results.all_split_points

    @staticmethod
    def _set_aim_rank(feature_name, split_point_array, missing_dict, total_counts, split_num):
        total_count = total_counts - missing_dict[feature_name]
        aim_ranks = [np.floor(x * total_count)
                     for x in np.linspace(0, 1, split_num)]
        aim_ranks = aim_ranks[1:]
        for idx, sp in enumerate(split_point_array):
            sp.set_aim_rank(aim_ranks[idx])
        return feature_name, split_point_array

    def _recursive_querying(self, summary_table):
        self.set_suffix(-1)
        query_points_table = self.init_query_points(summary_table.partitions,
                                                    split_num=self.params.bin_num + 1,
                                                    error_rank=self.error_rank,
                                                    need_first=False)

        f = functools.partial(self._set_aim_rank,
                              missing_dict=self.missing_counts,
                              total_counts=self.total_count,
                              split_num=self.params.bin_num + 1)
        query_points_table = query_points_table.map(f)
        global_ranks = self.query_values(summary_table, query_points_table)

        n_iter = 0
        while n_iter < self.params.max_iter:
            self.set_suffix(n_iter)
            query_points_table = query_points_table.join(global_ranks, self.renew_query_points_table)
            is_converge = self.check_converge(query_points_table)
            if self.role == consts.GUEST:
                self.transfer_variable.is_converge.remote(is_converge, suffix=self.suffix)
            LOGGER.debug(f"n_iter: {n_iter}, converged: {is_converge}")
            if is_converge:
                break
            global_ranks = self.query_values(summary_table, query_points_table)
            n_iter += 1
        return query_points_table

    @staticmethod
    def renew_query_points_table(query_points, ranks):
        assert len(query_points) == len(ranks)
        new_array = []

        for idx, node in enumerate(query_points):
            rank = ranks[idx]
            if node.fixed:
                new_node = copy.deepcopy(node)
            elif rank - node.aim_rank > node.allow_error_rank:
                new_node = node.create_left_new()
            elif node.aim_rank - rank > node.allow_error_rank:
                new_node = node.create_right_new()
            else:
                new_node = copy.deepcopy(node)
                new_node.fixed = True
            new_node.last_rank = rank
            new_array.append(new_node)
        return new_array

    @staticmethod
    def check_converge(query_table):
        def is_all_fixed(node_array):
            fix_array = [n.fixed for n in node_array]
            return functools.reduce(operator.and_, fix_array)
        fix_table = query_table.mapValues(is_all_fixed)
        return fix_table.reduce(operator.and_)
