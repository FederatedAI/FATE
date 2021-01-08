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
# import time
import uuid

from federatedml.feature.binning.base_binning import BaseBinning
from federatedml.feature.binning.quantile_summaries import quantile_summary_factory
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.statistic import data_overview
from federatedml.util import LOGGER
from federatedml.util import consts


class QuantileBinning(BaseBinning):
    """
    After quantile binning, the numbers of elements in each binning are equal.

    The result of this algorithm has the following deterministic bound:
    If the data_instances has N elements and if we request the quantile at probability `p` up to error
    `err`, then the algorithm will return a sample `x` from the data so that the *exact* rank
    of `x` is close to (p * N).
    More precisely,

    {{{
      floor((p - 2 * err) * N) <= rank(x) <= ceil((p + 2 * err) * N)
    }}}

    This method implements a variation of the Greenwald-Khanna algorithm (with some speed
    optimizations).
    """

    def __init__(self, params: FeatureBinningParam, abnormal_list=None, allow_duplicate=False):
        super(QuantileBinning, self).__init__(params, abnormal_list)
        self.summary_dict = None
        self.allow_duplicate = allow_duplicate

    def fit_split_points(self, data_instances):
        """
        Apply the binning method

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
                            ...                         # Other features
                            }
        """
        header = data_overview.get_header(data_instances)
        # LOGGER.debug("in _fit_split_point, cols_map: {}".format(self.bin_inner_param.bin_cols_map))

        self._default_setting(header)
        # self._init_cols(data_instances)
        percent_value = 1.0 / self.bin_num

        # calculate the split points
        percentile_rate = [i * percent_value for i in range(1, self.bin_num)]
        percentile_rate.append(1.0)
        is_sparse = data_overview.is_sparse_data(data_instances)

        self._fit_split_point(data_instances, is_sparse, percentile_rate)

        self.fit_category_features(data_instances)
        return self.bin_results.all_split_points

    @staticmethod
    def copy_merge(s1, s2):
        return s1.merge(s2)

    def _fit_split_point(self, data_instances, is_sparse, percentile_rate):
        if self.summary_dict is None:
            f = functools.partial(self.feature_summary,
                                  params=self.params,
                                  abnormal_list=self.abnormal_list,
                                  cols_dict=self.bin_inner_param.bin_cols_map,
                                  header=self.header,
                                  is_sparse=is_sparse)
            # summary_dict_table = data_instances.mapReducePartitions(f, self.copy_merge)
            summary_dict_table = data_instances.mapReducePartitions(f, lambda s1, s2: s1.merge(s2))
            # summary_dict = dict(summary_dict.collect())

            if is_sparse:
                total_count = data_instances.count()
                summary_dict_table = summary_dict_table.mapValues(lambda x: x.set_total_count(total_count))

            self.summary_dict = summary_dict_table
        else:
            summary_dict_table = self.summary_dict

        f = functools.partial(self._get_split_points,
                              allow_duplicate=self.allow_duplicate,
                              percentile_rate=percentile_rate)
        summary_dict = dict(summary_dict_table.mapValues(f).collect())

        for col_name, split_point in summary_dict.items():
            self.bin_results.put_col_split_points(col_name, split_point)

    @staticmethod
    def _get_split_points(summary, percentile_rate, allow_duplicate):
        split_point = []
        for percen_rate in percentile_rate:
            s_p = summary.query(percen_rate)
            if not allow_duplicate:
                if s_p not in split_point:
                    split_point.append(s_p)
            else:
                split_point.append(s_p)
        return split_point

    @staticmethod
    def feature_summary(data_iter, params, cols_dict, abnormal_list, header, is_sparse):
        summary_dict = {}

        summary_param = {'compress_thres': params.compress_thres,
                         'head_size': params.head_size,
                         'error': params.error,
                         'abnormal_list': abnormal_list}

        for col_name, col_index in cols_dict.items():
            quantile_summaries = quantile_summary_factory(is_sparse=is_sparse, param_dict=summary_param)
            summary_dict[col_name] = quantile_summaries
        _ = str(uuid.uuid1())
        for _, instant in data_iter:
            if not is_sparse:
                if type(instant).__name__ == 'Instance':
                    features = instant.features
                else:
                    features = instant
                for col_name, summary in summary_dict.items():
                    col_index = cols_dict[col_name]
                    summary.insert(features[col_index])
            else:
                data_generator = instant.features.get_all_data()
                for col_idx, col_value in data_generator:
                    col_name = header[col_idx]
                    if col_name not in cols_dict:
                        continue
                    summary = summary_dict[col_name]
                    summary.insert(col_value)
        result = []
        for features_name, summary_obj in summary_dict.items():
            summary_obj.compress()
            # result.append(((_, features_name), summary_obj))
            result.append((features_name, summary_obj))

        return result

    def query_quantile_point(self, query_points, col_names=None):

        if self.summary_dict is None:
            raise RuntimeError("Bin object should be fit before query quantile points")

        if col_names is None:
            col_names = self.bin_inner_param.bin_names

        summary_dict = self.summary_dict

        if isinstance(query_points, (int, float)):
            query_dict = {}
            for col_name in col_names:
                query_dict[col_name] = query_points
        elif isinstance(query_points, dict):
            query_dict = query_points
        else:
            raise ValueError("query_points has wrong type, should be a float, int or dict")

        result = {}
        for col_name, query_point in query_dict.items():
            summary = summary_dict[col_name]
            result[col_name] = summary.query(query_point)
        return result



