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

from arch.api.utils import log_utils
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.framework.homo.blocks import secure_mean_aggregator
from federatedml.framework.weights import DictWeights
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.util import abnormal_detection
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HomoFeatureBinningServer(object):
    def __init__(self):
        self.split_points_aggregator = secure_mean_aggregator.Server(enable_secure_aggregate=True)
        self.suffix = tuple()

    def set_suffix(self, suffix):
        self.suffix = suffix

    def average_run(self, data_instances=None, bin_param: FeatureBinningParam = None, bin_num=10, abnormal_list=None):
        agg_split_points = self.split_points_aggregator.mean_model(suffix=self.suffix)
        self.split_points_aggregator.send_aggregated_model(agg_split_points)


class HomoFeatureBinningClient(object):
    def __init__(self, bin_method=consts.QUANTILE):
        self.split_points_aggregator = secure_mean_aggregator.Client(enable_secure_aggregate=True)
        self.suffix = tuple()
        self.bin_method = bin_method
        self.bin_obj = None

    def set_suffix(self, suffix):
        self.suffix = suffix

    def average_run(self, data_instances, bin_param: FeatureBinningParam = None, bin_num=10, abnormal_list=None):
        if bin_param is None:
            bin_param = FeatureBinningParam(bin_num=bin_num)

        if self.bin_method == consts.QUANTILE:
            bin_obj = QuantileBinning(params=bin_param, abnormal_list=abnormal_list, allow_duplicate=True)
        else:
            raise ValueError("Homo Split Point do not accept bin_method: {}".format(self.bin_method))

        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

        split_points = bin_obj.fit_split_points(data_instances)
        split_points = {k: np.array(v) for k, v in split_points.items()}
        split_points_weights = DictWeights(d=split_points)

        self.split_points_aggregator.send_model(split_points_weights, self.suffix)
        dict_split_points = self.split_points_aggregator.get_aggregated_model(self.suffix)
        split_points = {k: list(v) for k, v in dict_split_points.unboxed.items()}
        self.bin_obj = bin_obj
        return split_points

    def convert_feature_to_bin(self, data_instances, split_points=None):
        if self.bin_obj is None:
            return None, None, None
        return self.bin_obj.convert_feature_to_bin(data_instances, split_points)
