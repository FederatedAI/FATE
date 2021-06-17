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

from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.statistic import data_overview
from federatedml.util import consts, LOGGER


class QuantileBinningTool(QuantileBinning):
    """
    Use for quantile binning data directly.
    """

    def __init__(self, bin_nums=consts.G_BIN_NUM, param_obj: FeatureBinningParam = None,
                 abnormal_list=None, allow_duplicate=False):
        if param_obj is None:
            param_obj = FeatureBinningParam(bin_num=bin_nums)
        super().__init__(params=param_obj, abnormal_list=abnormal_list, allow_duplicate=allow_duplicate)
        self.has_fit = False

    def fit_split_points(self, data_instances):
        res = super(QuantileBinningTool, self).fit_split_points(data_instances)
        self.has_fit = True
        return res

    def fit_summary(self, data_instances, is_sparse=None):
        if is_sparse is None:
            is_sparse = data_overview.is_sparse_data(data_instances)
            LOGGER.debug(f"is_sparse: {is_sparse}")

        f = functools.partial(self.feature_summary,
                              params=self.params,
                              abnormal_list=self.abnormal_list,
                              cols_dict=self.bin_inner_param.bin_cols_map,
                              header=self.header,
                              is_sparse=is_sparse)
        summary_dict_table = data_instances.mapReducePartitions(f, self.copy_merge)
        # summary_dict = dict(summary_dict.collect())

        if is_sparse:
            total_count = data_instances.count()
            summary_dict_table = summary_dict_table.mapValues(lambda x: x.set_total_count(total_count))
        return summary_dict_table

    def get_quantile_point(self, quantile):
        """
        Return the specific quantile point value

        Parameters
        ----------
        quantile : float, 0 <= quantile <= 1
            Specify which column(s) need to apply statistic.

        Returns
        -------
        return a dict of result quantile points.
        eg.
        quantile_point = {"x1": 3, "x2": 5... }
        """
        if not self.has_fit:
            raise RuntimeError("Quantile Binning Tool's split points should be fit before calling"
                               " get quantile points")

        f = functools.partial(self._get_split_points,
                              allow_duplicate=self.allow_duplicate,
                              percentile_rate=[quantile])
        quantile_points = dict(self.summary_dict.mapValues(f).collect())
        quantile_points = {k: v[0] for k, v in quantile_points.items()}
        return quantile_points

    def get_median(self):
        return self.get_quantile_point(0.5)
