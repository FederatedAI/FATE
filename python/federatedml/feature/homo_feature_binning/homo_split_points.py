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

import numpy as np

from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.framework.homo.blocks import secure_mean_aggregator
from federatedml.framework.weights import DictWeights
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.util import abnormal_detection
from federatedml.util import consts


class HomoFeatureBinningServer(object):
    def __init__(self):
        self.aggregator = secure_mean_aggregator.Server(enable_secure_aggregate=True)
        self.suffix = tuple()

    def set_suffix(self, suffix):
        self.suffix = suffix

    def average_run(self, data_instances=None, bin_param: FeatureBinningParam = None, bin_num=10, abnormal_list=None):
        agg_split_points = self.aggregator.mean_model(suffix=self.suffix)
        self.aggregator.send_aggregated_model(agg_split_points)

    def fit(self, *args, **kwargs):
        pass

    def query_quantile_points(self, data_instances, quantile_points):
        suffix = tuple(list(self.suffix) + [str(quantile_points)])
        agg_quantile_points = self.aggregator.mean_model(suffix=suffix)
        self.aggregator.send_aggregated_model(agg_quantile_points, suffix=suffix)


class HomoFeatureBinningClient(object):
    def __init__(self, bin_method=consts.QUANTILE):
        self.aggregator = secure_mean_aggregator.Client(enable_secure_aggregate=True)
        self.suffix = tuple()
        self.bin_method = bin_method
        self.bin_obj: QuantileBinning = None
        self.bin_param = None
        self.abnormal_list = None

    def set_suffix(self, suffix):
        self.suffix = suffix

    def average_run(self, data_instances, bin_num=10, abnormal_list=None):
        if self.bin_param is None:
            bin_param = FeatureBinningParam(bin_num=bin_num)
            self.bin_param = bin_param
        else:
            bin_param = self.bin_param

        if self.bin_method == consts.QUANTILE:
            bin_obj = QuantileBinning(params=bin_param, abnormal_list=abnormal_list, allow_duplicate=True)
        else:
            raise ValueError("Homo Split Point do not accept bin_method: {}".format(self.bin_method))

        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

        split_points = bin_obj.fit_split_points(data_instances)
        split_points = {k: np.array(v) for k, v in split_points.items()}
        split_points_weights = DictWeights(d=split_points)

        self.aggregator.send_model(split_points_weights, self.suffix)
        dict_split_points = self.aggregator.get_aggregated_model(self.suffix)
        split_points = {k: list(v) for k, v in dict_split_points.unboxed.items()}
        self.bin_obj = bin_obj
        return split_points

    def convert_feature_to_bin(self, data_instances, split_points=None):
        if self.bin_obj is None:
            return None, None, None
        return self.bin_obj.convert_feature_to_bin(data_instances, split_points)

    def set_bin_param(self, bin_param: FeatureBinningParam):
        if self.bin_param is not None:
            raise RuntimeError("Bin param has been set and it's immutable")
        self.bin_param = bin_param
        return self

    def set_abnormal_list(self, abnormal_list):
        self.abnormal_list = abnormal_list
        return self

    def fit(self, data_instances):
        if self.bin_obj is not None:
            return self

        if self.bin_param is None:
            self.bin_param = FeatureBinningParam()

        self.bin_obj = QuantileBinning(params=self.bin_param, abnormal_list=self.abnormal_list,
                                       allow_duplicate=True)
        self.bin_obj.fit_split_points(data_instances)
        return self

    def query_quantile_points(self, data_instances, quantile_points):
        if self.bin_obj is None:
            self.fit(data_instances)

        # bin_col_names = self.bin_obj.bin_inner_param.bin_names
        query_result = self.bin_obj.query_quantile_point(quantile_points)

        query_points = DictWeights(d=query_result)

        suffix = tuple(list(self.suffix) + [str(quantile_points)])
        self.aggregator.send_model(query_points, suffix)
        query_points = self.aggregator.get_aggregated_model(suffix)
        query_points = {k: v for k, v in query_points.unboxed.items()}
        return query_points
