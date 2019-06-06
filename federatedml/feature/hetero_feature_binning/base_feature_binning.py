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

from arch.api.model_manager import manager as model_manager
from arch.api.proto import feature_binning_meta_pb2, feature_binning_param_pb2
from arch.api.utils import log_utils
from federatedml.feature.binning.base_binning import IVAttributes
from federatedml.feature.binning.bucket_binning import BucketBinning
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.statistic.data_overview import get_header
from federatedml.util import abnormal_detection
from federatedml.util import consts
from federatedml.util.transfer_variable import HeteroFeatureBinningTransferVariable

LOGGER = log_utils.getLogger()


class BaseHeteroFeatureBinning(object):
    """
    Do binning method through guest and host

    Parameters
    ----------
    params : FeatureBinningParam
        Binning parameters set by users

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

    def __init__(self, params):
        self.bin_param = params

        self.transfer_variable = HeteroFeatureBinningTransferVariable()
        self.cols = params.cols
        self.cols_dict = {}
        self.binning_obj = None
        self.header = []
        self.has_synchronized = False
        self.flowid = ''
        self.binning_result = {}  # dict of iv_attr
        self.host_results = {}  # dict of host results
        self.party_name = 'Base'

    def _init_binning_obj(self):
        if self.bin_param.method == consts.QUANTILE:
            self.binning_obj = QuantileBinning(self.bin_param, self.party_name)
        elif self.bin_param.method == consts.BUCKET:
            self.binning_obj = BucketBinning(self.bin_param, self.party_name)
        else:
            # self.binning_obj = QuantileBinning(self.bin_param)
            raise ValueError("Binning method: {} is not supported yet".format(self.bin_param.method))

    def _save_meta(self, name, namespace):
        meta_protobuf_obj = feature_binning_meta_pb2.FeatureBinningMeta(
            method=self.bin_param.method,
            compress_thres=self.bin_param.compress_thres,
            head_size=self.bin_param.head_size,
            error=self.bin_param.error,
            bin_num=self.bin_param.bin_num,
            cols=self.cols,
            adjustment_factor=self.bin_param.adjustment_factor,
            local_only=self.bin_param.local_only)
        buffer_type = "HeteroFeatureBinning{}.meta".format(self.party_name)

        model_manager.save_model(buffer_type=buffer_type,
                                 proto_buffer=meta_protobuf_obj,
                                 name=name,
                                 namespace=namespace)
        return buffer_type

    def save_model(self, name, namespace, binning_result=None, host_results=None):

        if binning_result is None:
            binning_result = self.binning_result

        if host_results is None:
            host_results = self.host_results

        meta_buffer_type = self._save_meta(name, namespace)

        iv_attrs = {}
        for col_name, iv_attr in binning_result.items():
            iv_result = iv_attr.result_dict()
            iv_object = feature_binning_param_pb2.IVParam(**iv_result)

            iv_attrs[col_name] = iv_object
        binning_result_obj = feature_binning_param_pb2.FeatureBinningResult(binning_result=iv_attrs)

        final_host_results = {}
        for host_id, this_host_results in host_results.items():
            host_result = {}
            for col_name, iv_attr in this_host_results.items():
                iv_result = iv_attr.result_dict()
                iv_object = feature_binning_param_pb2.IVParam(**iv_result)
                host_result[col_name] = iv_object
            final_host_results[host_id] = feature_binning_param_pb2.FeatureBinningResult(binning_result=host_result)

        result_obj = feature_binning_param_pb2.FeatureBinningParam(binning_result=binning_result_obj,
                                                                   host_results=final_host_results)

        param_buffer_type = "HeteroFeatureBinning{}.param".format(self.party_name)

        model_manager.save_model(buffer_type=param_buffer_type,
                                 proto_buffer=result_obj,
                                 name=name,
                                 namespace=namespace)

        return [(meta_buffer_type, param_buffer_type)]

    def load_model(self, name, namespace):

        result_obj = feature_binning_param_pb2.FeatureBinningParam()
        return_code = model_manager.read_model(buffer_type="HeteroFeatureBinning{}.param".format(self.party_name),
                                               proto_buffer=result_obj,
                                               name=name,
                                               namespace=namespace)
        binning_result_obj = dict(result_obj.binning_result.binning_result)
        host_params = dict(result_obj.host_results)
        # LOGGER.debug("Party name is :{}".format(self.party_name))
        # LOGGER.debug('Loading model, binning_result_obj is : {}'.format(binning_result_obj))
        self.binning_result = {}
        self.host_results = {}
        self.cols = []
        for col_name, iv_attr_obj in binning_result_obj.items():
            iv_attr = IVAttributes([], [], [], [], [], [])
            iv_attr.reconstruct(iv_attr_obj)
            self.binning_result[col_name] = iv_attr
            self.cols.append(col_name)

        for host_name, host_result_obj in host_params.items():
            host_result_obj = dict(host_result_obj.binning_result)
            for col_name, iv_attr_obj in host_result_obj.items():
                iv_attr = IVAttributes([], [], [], [], [], [])
                iv_attr.reconstruct(iv_attr_obj)
                host_result_obj[col_name] = iv_attr
            self.host_results[host_name] = host_result_obj
        return return_code

    def set_flowid(self, flowid="samole"):
        self.flowid = flowid
        self.transfer_variable.set_flowid(self.flowid)

    def _parse_cols(self, data_instances):
        if self.header is not None and len(self.header) != 0:
            return
        header = get_header(data_instances)
        self.header = header
        # LOGGER.debug("data_instance count: {}, header: {}".format(data_instances.count(), header))
        if self.cols == -1:
            if header is None:
                raise RuntimeError('Cannot get feature header, please check input data')
            self.cols = header
        self.cols_dict = {}
        for col in self.cols:
            col_index = header.index(col)
            self.cols_dict[col] = col_index

    def set_schema(self, data_instance):
        data_instance.schema = {"header": self.header}

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
