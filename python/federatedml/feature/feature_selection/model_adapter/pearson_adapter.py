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

from federatedml.feature.feature_selection.model_adapter import isometric_model
from federatedml.feature.feature_selection.model_adapter.adapter_base import BaseAdapter
from federatedml.util import consts


class PearsonMetricInfo(object):
    def __init__(self, local_corr, col_names, corr=None, host_col_names=None, parties=None):
        self.local_corr = local_corr
        self.col_names = col_names
        self.corr = corr
        self.host_col_names = host_col_names
        self.parties = parties

    @property
    def host_party_id(self):
        assert isinstance(self.parties, list) and len(self.parties) == 2
        return self.parties[1][1]


class PearsonAdapter(BaseAdapter):

    def convert(self, model_meta, model_param):
        local_vif = model_param.local_vif
        col_names = list(model_param.names)
        local_corr = np.array(model_param.local_corr).reshape(model_param.shape, model_param.shape)

        from federatedml.util import LOGGER
        for idx in range(local_corr.shape[0]):
            corr_col = local_corr[idx, :]
            # LOGGER.debug(f"local_col_idx: {idx}, corr_col: {corr_col}")

        if model_param.corr:
            corr = np.array(model_param.corr).reshape(*model_param.shapes)

            for idx in range(corr.shape[1]):
                corr_col = corr[:, idx]
                # LOGGER.debug(f"col_idx: {idx}, corr_col: {corr_col}")

            host_names = list(list(model_param.all_names)[1].names)
            parties = list(model_param.parties)
        else:
            corr = None
            host_names = None
            parties = None
        pearson_metric = PearsonMetricInfo(local_corr=local_corr, col_names=col_names,
                                           corr=corr, host_col_names=host_names, parties=parties)

        single_info = isometric_model.SingleMetricInfo(
            values=local_vif,
            col_names=col_names
        )
        result = isometric_model.IsometricModel()
        result.add_metric_value(metric_name=consts.VIF, metric_info=single_info)
        result.add_metric_value(metric_name=consts.PEARSON, metric_info=pearson_metric)
        return result
