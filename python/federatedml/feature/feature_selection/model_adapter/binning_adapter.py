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
import operator

from federatedml.feature.feature_selection.model_adapter import isometric_model
from federatedml.feature.feature_selection.model_adapter.adapter_base import BaseAdapter
from federatedml.util import LOGGER
from federatedml.util import consts


class BinningAdapter(BaseAdapter):

    def _load_one_class(self, local_result, remote_results):
        values_dict = dict(local_result.binning_result)
        values_sorted_dict = sorted(values_dict.items(), key=operator.itemgetter(0))
        values = []
        col_names = []
        for n, v in values_sorted_dict:
            values.append(v.iv)
            col_names.append(n)
        # LOGGER.debug(f"When loading iv, values: {values}, col_names: {col_names}")
        host_party_ids = [int(x.party_id) for x in remote_results]
        host_values = []
        host_col_names = []
        for host_obj in remote_results:
            binning_result = dict(host_obj.binning_result)
            h_values = []
            h_col_names = []
            for n, v in binning_result.items():
                h_values.append(v.iv)
                h_col_names.append(n)
            host_values.append(np.array(h_values))
            host_col_names.append(h_col_names)
        # LOGGER.debug(f"host_party_ids: {host_party_ids}, host_values: {host_values},"
        #             f"host_col_names: {host_col_names}")
        LOGGER.debug(f"host_party_ids: {host_party_ids}")
        single_info = isometric_model.SingleMetricInfo(
            values=np.array(values),
            col_names=col_names,
            host_party_ids=host_party_ids,
            host_values=host_values,
            host_col_names=host_col_names
        )
        return single_info

    def convert(self, model_meta, model_param):

        multi_class_result = model_param.multi_class_result
        has_remote_result = multi_class_result.has_host_result
        label_counts = len(list(multi_class_result.labels))
        local_results = list(multi_class_result.results)
        host_results = list(multi_class_result.host_results)

        result = isometric_model.IsometricModel()
        for idx, lr in enumerate(local_results):
            if label_counts == 2:
                result.add_metric_value(metric_name=f"iv",
                                        metric_info=self._load_one_class(lr, host_results))
            else:
                if has_remote_result:
                    remote_results = [hs for i, hs in enumerate(host_results) if (i % label_counts) == idx]
                else:
                    remote_results = []
                result.add_metric_value(metric_name=f"iv",
                                        metric_info=self._load_one_class(lr, remote_results))
        return result
