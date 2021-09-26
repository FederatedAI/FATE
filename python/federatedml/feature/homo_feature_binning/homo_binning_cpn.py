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

from federatedml.model_base import ModelBase
from federatedml.param.feature_binning_param import HomoFeatureBinningParam
from federatedml.feature.homo_feature_binning import virtual_summary_binning, recursive_query_binning
from federatedml.util import consts
from federatedml.feature.hetero_feature_binning.base_feature_binning import BaseFeatureBinning
from federatedml.transfer_variable.transfer_class.homo_binning_transfer_variable import HomoBinningTransferVariable


class HomoBinningArbiter(BaseFeatureBinning):
    def __init__(self):
        super().__init__()
        self.binning_obj = None
        self.transfer_variable = HomoBinningTransferVariable()
        self.model_param = HomoFeatureBinningParam()

    def _init_model(self, model_param):
        self.model_param = model_param
        if self.model_param.method == consts.VIRTUAL_SUMMARY:
            self.binning_obj = virtual_summary_binning.Server(self.model_param)
        elif self.model_param.method == consts.RECURSIVE_QUERY:
            self.binning_obj = recursive_query_binning.Server(self.model_param)
        else:
            raise ValueError(f"Method: {self.model_param.method} cannot be recognized")

    def fit(self, *args):
        self.binning_obj.set_transfer_variable(self.transfer_variable)
        self.binning_obj.fit_split_points()

    def transform(self, data_instances):
        pass


class HomoBinningClient(BaseFeatureBinning):
    def __init__(self):
        super().__init__()
        self.binning_obj = None
        self.transfer_variable = HomoBinningTransferVariable()
        self.model_param = HomoFeatureBinningParam()

    def _init_model(self, model_param: HomoFeatureBinningParam):
        self.transform_type = self.model_param.transform_param.transform_type

        self.model_param = model_param
        if self.model_param.method == consts.VIRTUAL_SUMMARY:
            self.binning_obj = virtual_summary_binning.Client(self.model_param)
        elif self.model_param.method == consts.RECURSIVE_QUERY:
            self.binning_obj = recursive_query_binning.Client(role=self.component_properties.role,
                                                              params=self.model_param
                                                              )
        else:
            raise ValueError(f"Method: {self.model_param.method} cannot be recognized")

    def fit(self, data_instances):
        self._abnormal_detection(data_instances)
        self._setup_bin_inner_param(data_instances, self.model_param)
        transformed_instances = data_instances.mapValues(self.data_format_transform)
        transformed_instances.schema = self.schema
        self.binning_obj.set_bin_inner_param(self.bin_inner_param)
        self.binning_obj.set_transfer_variable(self.transfer_variable)
        split_points = self.binning_obj.fit_split_points(transformed_instances)
        data_out = self.transform(data_instances)
        summary = {}
        for k, v in split_points.items():
            summary[k] = list(v)
        self.set_summary({"split_points": summary})
        return data_out
