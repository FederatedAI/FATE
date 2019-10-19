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

from federatedml.protobuf.generated import feature_binning_meta_pb2, feature_binning_param_pb2
from arch.api.utils import log_utils
from federatedml.feature.binning.base_binning import IVAttributes
from federatedml.feature.binning.bucket_binning import BucketBinning
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.model_base import ModelBase
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.statistic.data_overview import get_header
from federatedml.util import abnormal_detection
from federatedml.util import consts
from federatedml.transfer_variable.transfer_class.hetero_feature_binning_transfer_variable import \
    HeteroFeatureBinningTransferVariable

LOGGER = log_utils.getLogger()

MODEL_PARAM_NAME = 'FeatureBinningParam'
MODEL_META_NAME = 'FeatureBinningMeta'


class BaseHeteroFeatureBinning(ModelBase):
    """
    Do binning method through guest and host

    Attributes
    ----------
    header : list
        record headers of input table

    has_synchronized : bool
        Record whether the encryption information has been synchronized or not.

    flowid : str
        Use in cross validation

    binning_result: dict
        Record binning result of guest party. The format is {'col_name': 'iv_attr', ... }

    host_results: dict
        This attribute uses to record host results. For future version which may record multiple host results,
        the format is dict of dict.
        e.g.
        host_results = {'host1': {'x1': iv1, 'x2: iv2}
                        'host2': ...
                        }

    """

    def __init__(self):
        super(BaseHeteroFeatureBinning, self).__init__()
        self.transfer_variable = HeteroFeatureBinningTransferVariable()
        self.cols = []
        self.cols_dict = {}
        self.binning_obj = None
        self.header = []
        self.schema = {}
        self.has_synchronized = False
        self.flowid = ''
        self.binning_result = {}  # dict of iv_attr
        self.host_results = {}  # dict of host results
        self.party_name = 'Base'
        self.model_param = FeatureBinningParam()
        self.transform_cols_idx = []
        self.transform_type = ''

    def _init_model(self, params):
        self.model_param = params
        self.cols_index = params.cols
        self.transform_cols_idx = self.model_param.transform_param.transform_cols
        self.transform_type = self.model_param.transform_param.transform_type

        if self.model_param.method == consts.QUANTILE:
            self.binning_obj = QuantileBinning(self.model_param, self.party_name)
        elif self.model_param.method == consts.BUCKET:
            self.binning_obj = BucketBinning(self.model_param, self.party_name)
        else:
            # self.binning_obj = QuantileBinning(self.bin_param)
            raise ValueError("Binning method: {} is not supported yet".format(self.model_param.method))

    def transform(self, data_instances):
        self._parse_cols(data_instances)
        data_instances = self.binning_obj.transform(data_instances, self.transform_cols_idx, self.transform_type)
        self.set_schema(data_instances)
        self.data_output = data_instances

        return data_instances

    def _get_meta(self):
        col_list = [str(x) for x in self.cols]

        transform_param = feature_binning_meta_pb2.TransformMeta(
            transform_cols=self.transform_cols_idx,
            transform_type=self.model_param.transform_param.transform_type
        )

        meta_protobuf_obj = feature_binning_meta_pb2.FeatureBinningMeta(
            method=self.model_param.method,
            compress_thres=self.model_param.compress_thres,
            head_size=self.model_param.head_size,
            error=self.model_param.error,
            bin_num=self.model_param.bin_num,
            cols=col_list,
            adjustment_factor=self.model_param.adjustment_factor,
            local_only=self.model_param.local_only,
            need_run=self.need_run,
            transform_param=transform_param
        )
        return meta_protobuf_obj

    def _get_param(self):

        binning_result = self.binning_result

        host_results = self.host_results

        iv_attrs = {}
        for col_name, iv_attr in binning_result.items():
            iv_result = iv_attr.result_dict()
            iv_object = feature_binning_param_pb2.IVParam(**iv_result)
            iv_attrs[col_name] = iv_object
        binning_result_obj = feature_binning_param_pb2.FeatureBinningResult(binning_result=iv_attrs)

        final_host_results = {}
        for host_id, this_host_results in host_results.items():
            host_result = {}
            for host_col_idx, iv_attr in this_host_results.items():
                iv_result = iv_attr.result_dict()
                iv_object = feature_binning_param_pb2.IVParam(**iv_result)
                host_result[str(host_col_idx)] = iv_object
            final_host_results[host_id] = feature_binning_param_pb2.FeatureBinningResult(binning_result=host_result)

        result_obj = feature_binning_param_pb2.FeatureBinningParam(binning_result=binning_result_obj,
                                                                   host_results=final_host_results,
                                                                   header=self.header)
        # json_result = json_format.MessageToJson(result_obj)
        # LOGGER.debug("json_result: {}".format(json_result))
        return result_obj

    def _load_model(self, model_dict):
        model_param = list(model_dict.get('model').values())[0].get(MODEL_PARAM_NAME)
        model_meta = list(model_dict.get('model').values())[0].get(MODEL_META_NAME)

        assert isinstance(model_meta, feature_binning_meta_pb2.FeatureBinningMeta)
        self.transform_cols_idx = list(model_meta.transform_param.transform_cols)
        self.transform_type = model_meta.transform_param.transform_type

        self.cols = list(map(int, model_meta.cols))
        bin_method = str(model_meta.method)
        if bin_method == consts.QUANTILE:
            self.binning_obj = QuantileBinning(model_meta, self.party_name)
        else:
            self.binning_obj = BucketBinning(model_meta, self.party_name)

        binning_result_obj = dict(model_param.binning_result.binning_result)
        host_params = dict(model_param.host_results)

        self.binning_result = {}
        self.host_results = {}
        for col_name, iv_attr_obj in binning_result_obj.items():
            iv_attr = IVAttributes([], [], [], [], [], [])
            iv_attr.reconstruct(iv_attr_obj)
            self.binning_obj.reconstruct_by_iv_obj(col_name, iv_attr)
            self.binning_result[col_name] = iv_attr
            # self.cols.append(col_name)

        for host_name, host_result_obj in host_params.items():
            host_result_obj = dict(host_result_obj.binning_result)
            for col_name, iv_attr_obj in host_result_obj.items():
                iv_attr = IVAttributes([], [], [], [], [], [])
                iv_attr.reconstruct(iv_attr_obj)
                host_result_obj[col_name] = iv_attr
            self.host_results[host_name] = host_result_obj
        # LOGGER.debug("In feature binning load model, self.binning_result: {}, cols: {}, host_results: {}".format(
        #     self.binning_result, self.cols, self.host_results
        # ))

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

    def _parse_cols(self, data_instances):
        if self.header is not None and len(self.header) != 0:
            return

        LOGGER.debug("Before Binning, schema is : {}".format(data_instances.schema))
        header = get_header(data_instances)
        self.schema = data_instances.schema
        self.header = header

        # LOGGER.debug("data_instance count: {}, header: {}".format(data_instances.count(), header))
        if self.cols_index == -1:
            if header is None:
                raise RuntimeError('Cannot get feature header, please check input data')
            self.cols = [i for i in range(len(header))]
        else:
            self.cols = self.cols_index

        if self.transform_cols_idx == -1:
            self.transform_cols_idx = self.cols

        self.cols_dict = {}
        for col in self.cols:
            col_name = header[col]
            self.cols_dict[col_name] = col

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
