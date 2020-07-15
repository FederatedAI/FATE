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

from federatedml.feature.feature_selection.model_adaptor import isometric_model
from federatedml.feature.feature_selection.model_adaptor.adapter_base import BaseAdapter
from federatedml.util import consts
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class BinningAdapter(BaseAdapter):

    def convert(self, model_meta, model_param):
        values_dict = dict(model_param.binning_result.binning_result)
        values = []
        col_names = []
        for n, v in values_dict.items():
            values.append(v.iv)
            col_names.append(n)
        host_results = list(model_param.host_results)
        LOGGER.debug(f"In binning adapter convert, host_results: {host_results}")
        host_party_ids = [int(x.party_id) for x in host_results]
        host_values = []
        host_col_names = []
        for host_obj in host_results:
            binning_result = dict(host_obj.binning_result)
            h_values = []
            h_col_names = []
            for n, v in binning_result.items():
                h_values.append(v.iv)
                h_col_names.append(n)
            host_values.append(np.array(h_values))
            host_col_names.append(h_col_names)
        LOGGER.debug(f"host_party_ids: {host_party_ids}, host_values: {host_values},"
                     f"host_col_names: {host_col_names}")
        single_info = isometric_model.SingleMetricInfo(
            values=np.array(values),
            col_names=col_names,
            host_party_ids=host_party_ids,
            host_values=host_values,
            host_col_names=host_col_names
        )
        result = isometric_model.IsometricModel()
        result.add_metric_value(metric_name=consts.IV, metric_info=single_info)
        return result
