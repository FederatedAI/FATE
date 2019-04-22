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

import copy
import functools

import numpy as np

from arch.api.model_manager import manager as model_manager
from arch.api.proto import feature_selection_meta_pb2, feature_selection_param_pb2
from arch.api.proto.feature_binning_meta_pb2 import FeatureBinningMeta
from federatedml.statistic.data_overview import get_features_shape
from federatedml.util import abnormal_detection
from federatedml.util.transfer_variable import HeteroFeatureSelectionTransferVariable


class BaseHeteroFeatureSelection(object):
    def __init__(self, params):
        self.params = params
        self.transfer_variable = HeteroFeatureSelectionTransferVariable()
        self.cols = params.select_cols
        self.filter_method = params.filter_method
        self.header = []

    def _save_meta(self, name, namespace):
        unique_param_dict = copy.deepcopy(self.params.unique_param.__dict__)

        unique_param = feature_selection_meta_pb2.UniqueValueParam(**unique_param_dict)

        iv_dict = copy.deepcopy(self.params.iv_param.__dict__)
        bin_dict = copy.deepcopy(self.params.iv_param.bin_param.__dict__)
        del bin_dict['process_method']
        del bin_dict['result_table']
        del bin_dict['result_namespace']
        del bin_dict['display_result']
        if bin_dict['cols'] == -1:
            bin_dict['cols'] = self.cols
        bin_param = FeatureBinningMeta()
        iv_dict["bin_param"] = bin_param

        iv_param = feature_selection_meta_pb2.IVSelectionParam(**iv_dict)
        coe_param_dict = copy.deepcopy(self.params.coe_param.__dict__)
        coe_param = feature_selection_meta_pb2.CoeffOfVarSelectionParam(**coe_param_dict)
        outlier_param_dict = copy.deepcopy(self.params.outlier_param.__dict__)

        outlier_param = feature_selection_meta_pb2.OutlierColsSelectionParam(**outlier_param_dict)

        meta_protobuf_obj = feature_selection_meta_pb2.FeatureSelectionMeta(filter_methods=self.filter_method,
                                                                            local_only=self.params.local_only,
                                                                            select_cols=self.header,
                                                                            unique_param=unique_param,
                                                                            iv_param=iv_param,
                                                                            coe_param=coe_param,
                                                                            outlier_param=outlier_param)
        buffer_type = "HeteroFeatureSelectionGuest.meta"

        model_manager.save_model(buffer_type=buffer_type,
                                 proto_buffer=meta_protobuf_obj,
                                 name=name,
                                 namespace=namespace)
        return buffer_type

    def save_model(self, name, namespace):
        meta_buffer_type = self._save_meta(name, namespace)

        result_obj = feature_selection_param_pb2.FeatureSelectionParam(results=self.results)
        param_buffer_type = "HeteroFeatureSelectionGuest.param"

        model_manager.save_model(buffer_type=param_buffer_type,
                                 proto_buffer=result_obj,
                                 name=name,
                                 namespace=namespace)
        return [(meta_buffer_type, param_buffer_type)]

    def load_model(self, name, namespace):
        result_obj = feature_selection_param_pb2.FeatureSelectionParam()
        model_manager.read_model(buffer_type="HeteroFeatureSelectionGuest.param",
                                 proto_buffer=result_obj,
                                 name=name,
                                 namespace=namespace)

        self.results = list(result_obj.results)
        if len(self.results) == 0:
            self.left_cols = -1
        else:
            result_obj = self.results[-1]
            self.left_cols = list(result_obj.left_cols)

    @staticmethod
    def select_cols(instance, left_cols):
        new_feature = []
        for col in left_cols:
            new_feature.append(instance.features[col])
        new_feature = np.array(new_feature)
        instance.features = new_feature
        return instance

    def _reset_header(self):
        """
        The cols and left_cols record the index of header. Replace header based on the change
        between left_cols and cols.
        """
        new_header = []
        for col in self.left_cols:
            idx = self.cols.index(col)
            new_header.append(self.header[idx])
        self.header = new_header

    def _transfer_data(self, data_instances):
        if self.left_cols == -1:
            features_shape = get_features_shape(data_instances)
            if features_shape is None:
                raise RuntimeError('Cannot get feature shape, please check input data')
            self.left_cols = [i for i in range(features_shape)]

        f = functools.partial(self.select_cols,
                              left_cols=self.left_cols)

        new_data = data_instances.mapValues(f)
        self._reset_header()
        return new_data

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

    def set_flowid(self, flowid="samole"):
        self.flowid = flowid
        self.transfer_variable.set_flowid(self.flowid)
