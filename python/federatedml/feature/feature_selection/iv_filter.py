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

from federatedml.feature.feature_selection.iso_model_filter import FederatedIsoModelFilter
from federatedml.feature.feature_selection.model_adapter import isometric_model
from federatedml.param.feature_selection_param import IVFilterParam
from federatedml.util import LOGGER


class IVFilter(FederatedIsoModelFilter):

    def _parse_filter_param(self, filter_param: IVFilterParam):
        super()._parse_filter_param(filter_param)
        self.merge_type = filter_param.mul_class_merge_type[0]

    def _merge_iv(self):
        metric_infos = self.iso_model.get_all_metric_info()
        col_names = metric_infos[0].col_names
        host_party_ids = metric_infos[0].host_party_ids
        host_col_names = metric_infos[0].host_col_names

        values = metric_infos[0].values
        host_values = np.array(metric_infos[0].host_values)
        if self.merge_type == "max":
            for m in metric_infos[1:]:
                values = np.maximum(values, m.values)
                host_values = np.maximum(host_values, m.host_values)
        elif self.merge_type == "min":
            for m in metric_infos[1:]:
                values = np.maximum(values, m.values)
                host_values = np.maximum(host_values, m.host_values)
        else:
            for m in metric_infos[1:]:
                values += m.values
                host_values += m.host_values

        """for m in metric_infos[1:]:
            if self.merge_type == "max":
                values = np.maximum(values, m.values)
                host_values = np.maximum(host_values, m.host_values)
            elif self.merge_type == "min":
                values = np.minimum(values, m.values)
                host_values = np.minimum(host_values, m.host_values)
            else:
                values += m.values
                host_values += m.host_values
        """
        if self.merge_type == 'average':
            values /= len(metric_infos)
            host_values /= len(metric_infos)
        # LOGGER.debug(f"After merge, iv_values: {values}, host_values: {host_values},"
        #              f" merge_type:{self.merge_type}")
        single_info = isometric_model.SingleMetricInfo(
            values=values,
            col_names=col_names,
            host_party_ids=host_party_ids,
            host_values=host_values,
            host_col_names=host_col_names
        )
        return single_info

    def _guest_fit(self, suffix):
        # for idx, m in enumerate(self.metrics):
        value_obj = self._merge_iv()
        self._fix_with_value_obj(value_obj, suffix)
