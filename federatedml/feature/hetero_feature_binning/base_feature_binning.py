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

from arch.api.utils import log_utils
from federatedml.feature.binning.base_binning import BaseBinning
from federatedml.feature.binning.bin_inner_param import BinInnerParam
from federatedml.feature.binning.bucket_binning import BucketBinning
from federatedml.feature.binning.optimal_binning.optimal_binning import OptimalBinning
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.model_base import ModelBase
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.protobuf.generated import feature_binning_meta_pb2, feature_binning_param_pb2
from federatedml.statistic.data_overview import get_header
from federatedml.transfer_variable.transfer_class.hetero_feature_binning_transfer_variable import \
    HeteroFeatureBinningTransferVariable
from federatedml.util import abnormal_detection
from federatedml.util import consts

LOGGER = log_utils.getLogger()

MODEL_PARAM_NAME = 'FeatureBinningParam'
MODEL_META_NAME = 'FeatureBinningMeta'


class BaseHeteroFeatureBinning(ModelBase):
    """
    Do binning method through guest and host

    """

    def __init__(self):
        super(BaseHeteroFeatureBinning, self).__init__()
        self.transfer_variable = HeteroFeatureBinningTransferVariable()
        self.binning_obj: BaseBinning = None
        self.header = None
        self.schema = None
        self.host_results = []
        self.transform_type = None

        self.model_param = FeatureBinningParam()
        self.bin_inner_param = BinInnerParam()

    def _init_model(self, params: FeatureBinningParam):
        self.model_param = params

        self.transform_type = self.model_param.transform_param.transform_type

        if self.model_param.method == consts.QUANTILE:
            self.binning_obj = QuantileBinning(self.model_param)
        elif self.model_param.method == consts.BUCKET:
            self.binning_obj = BucketBinning(self.model_param)
        elif self.model_param.method == consts.OPTIMAL:
            if self.role == consts.HOST:
                self.model_param.bin_num = self.model_param.optimal_binning_param.init_bin_nums
                self.binning_obj = QuantileBinning(self.model_param)
            else:
                self.binning_obj = OptimalBinning(self.model_param)
        else:
            # self.binning_obj = QuantileBinning(self.bin_param)
            raise ValueError("Binning method: {} is not supported yet".format(self.model_param.method))
        LOGGER.debug("in _init_model, role: {}, local_partyid: {}".format(self.role, self.component_properties))
        self.binning_obj.set_role_party(self.role, self.component_properties.local_partyid)

    def _setup_bin_inner_param(self, data_instances, params: FeatureBinningParam):
        if self.schema is not None:
            return

        self.header = get_header(data_instances)
        self.schema = data_instances.schema
        self.bin_inner_param.set_header(self.header)
        if params.bin_indexes == -1:
            self.bin_inner_param.set_bin_all()
        else:
            self.bin_inner_param.add_bin_indexes(params.bin_indexes)
            self.bin_inner_param.add_bin_names(params.bin_names)

        self.bin_inner_param.add_category_indexes(params.category_indexes)
        self.bin_inner_param.add_category_names(params.category_names)

        if params.transform_param.transform_cols == -1:
            self.bin_inner_param.set_transform_all()
        else:
            self.bin_inner_param.add_transform_bin_indexes(params.transform_param.transform_cols)
            self.bin_inner_param.add_transform_bin_names(params.transform_param.transform_names)

        self.binning_obj.set_bin_inner_param(self.bin_inner_param)

    def transform(self, data_instances):
        self._setup_bin_inner_param(data_instances, self.model_param)
        data_instances = self.binning_obj.transform(data_instances, self.transform_type)
        self.set_schema(data_instances)
        self.data_output = data_instances
        return data_instances

    def _get_meta(self):
        # col_list = [str(x) for x in self.cols]

        transform_param = feature_binning_meta_pb2.TransformMeta(
            transform_cols=self.bin_inner_param.transform_bin_indexes,
            transform_type=self.model_param.transform_param.transform_type
        )

        meta_protobuf_obj = feature_binning_meta_pb2.FeatureBinningMeta(
            method=self.model_param.method,
            compress_thres=self.model_param.compress_thres,
            head_size=self.model_param.head_size,
            error=self.model_param.error,
            bin_num=self.model_param.bin_num,
            cols=self.bin_inner_param.bin_names,
            adjustment_factor=self.model_param.adjustment_factor,
            local_only=self.model_param.local_only,
            need_run=self.need_run,
            transform_param=transform_param
        )
        return meta_protobuf_obj

    def _get_param(self):
        binning_result_obj = self.binning_obj.bin_results.generated_pb()
        # binning_result_obj = self.bin_results.generated_pb()
        host_results = [x.bin_results.generated_pb() for x in self.host_results]

        result_obj = feature_binning_param_pb2.FeatureBinningParam(binning_result=binning_result_obj,
                                                                   host_results=host_results,
                                                                   header=self.header)
        # json_result = json_format.MessageToJson(result_obj)
        # LOGGER.debug("json_result: {}".format(json_result))
        return result_obj

    def load_model(self, model_dict):
        model_param = list(model_dict.get('model').values())[0].get(MODEL_PARAM_NAME)
        model_meta = list(model_dict.get('model').values())[0].get(MODEL_META_NAME)

        self.bin_inner_param = BinInnerParam()

        assert isinstance(model_meta, feature_binning_meta_pb2.FeatureBinningMeta)
        assert isinstance(model_param, feature_binning_param_pb2.FeatureBinningParam)

        self.header = list(model_param.header)
        self.bin_inner_param.set_header(self.header)

        self.bin_inner_param.add_transform_bin_indexes(list(model_meta.transform_param.transform_cols))
        self.bin_inner_param.add_bin_names(list(model_meta.cols))
        self.transform_type = model_meta.transform_param.transform_type

        bin_method = str(model_meta.method)
        if bin_method == consts.QUANTILE:
            self.binning_obj = QuantileBinning(params=model_meta)
        else:
            self.binning_obj = BucketBinning(params=model_meta)

        self.binning_obj.set_role_party(self.role, self.component_properties.local_partyid)
        self.binning_obj.set_bin_inner_param(self.bin_inner_param)
        self.binning_obj.bin_results.reconstruct(model_param.binning_result)

        self.host_results = []
        for host_pb in model_param.host_results:
            host_bin_obj = BaseBinning()
            host_bin_obj.bin_results.reconstruct(host_pb)
            self.host_results.append(host_bin_obj)

    def export_model(self):
        if self.model_output is not None:
            return self.model_output

        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            MODEL_META_NAME: meta_obj,
            MODEL_PARAM_NAME: param_obj
        }
        self.model_output = result
        return result

    def save_data(self):
        return self.data_output

    def set_schema(self, data_instance):
        self.schema['header'] = self.header
        data_instance.schema = self.schema
        LOGGER.debug("After Binning, when setting schema, schema is : {}".format(data_instance.schema))

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
