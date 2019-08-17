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

import functools

import numpy as np

from arch.api.model_manager import manager as model_manager
from arch.api.proto import feature_selection_meta_pb2, feature_selection_param_pb2
from federatedml.statistic.data_overview import get_header
from federatedml.util import abnormal_detection
from federatedml.util.transfer_variable import HeteroFeatureSelectionTransferVariable
from federatedml.param.param import FeatureBinningParam
from federatedml.feature.hetero_feature_binning.hetero_binning_guest import HeteroFeatureBinningGuest
from federatedml.feature.hetero_feature_binning.hetero_binning_host import HeteroFeatureBinningHost
from federatedml.util import consts


class BaseHeteroFeatureSelection(object):
    def __init__(self, params):
        self.params = params
        self.transfer_variable = HeteroFeatureSelectionTransferVariable()
        self.cols = params.select_cols
        self.left_col_names = []  # temp result
        self.left_cols = {}  # final result
        self.cols_dict = {}
        self.filter_method = params.filter_method
        self.header = []
        self.party_name = 'Base'

        self.filter_meta_list = []
        self.filter_param_list = []

        # Possible previous model
        self.binning_model = None

        # All possible meta
        self.unique_meta = None
        self.iv_value_meta = None
        self.iv_percentile_meta = None
        self.coe_meta = None
        self.outlier_meta = None

        # Use to save each model's result
        self.results = []

    def init_previous_model(self, **models):
        if 'binning_model' in models:
            binning_model_params = models.get('binning_model')
            binning_param = FeatureBinningParam()
            if self.party_name == consts.GUEST:
                binning_obj = HeteroFeatureBinningGuest(binning_param)
            else:
                binning_obj = HeteroFeatureBinningHost(binning_param)

            name = binning_model_params.get('name')
            namespace = binning_model_params.get('namespace')

            binning_obj.load_model(name, namespace)
            self.binning_model = binning_obj

    def _save_meta(self, name, namespace):

        meta_protobuf_obj = feature_selection_meta_pb2.FeatureSelectionMeta(filter_methods=self.filter_method,
                                                                            local_only=self.params.local_only,
                                                                            cols=self.cols,
                                                                            unique_meta=self.unique_meta,
                                                                            iv_value_meta=self.iv_value_meta,
                                                                            iv_percentile_meta=self.iv_percentile_meta,
                                                                            coe_meta=self.coe_meta,
                                                                            outlier_meta=self.outlier_meta)
        buffer_type = "HeteroFeatureSelection{}.meta".format(self.party_name)

        model_manager.save_model(buffer_type=buffer_type,
                                 proto_buffer=meta_protobuf_obj,
                                 name=name,
                                 namespace=namespace)
        return buffer_type

    def save_model(self, name, namespace):
        meta_buffer_type = self._save_meta(name, namespace)

        left_col_obj = feature_selection_param_pb2.LeftCols(original_cols=self.cols,
                                                            left_cols=self.left_cols)

        result_obj = feature_selection_param_pb2.FeatureSelectionParam(results=self.results,
                                                                       final_left_cols=left_col_obj)

        param_buffer_type = "HeteroFeatureSelection{}.param".format(self.party_name)

        model_manager.save_model(buffer_type=param_buffer_type,
                                 proto_buffer=result_obj,
                                 name=name,
                                 namespace=namespace)
        return [(meta_buffer_type, param_buffer_type)]

    def load_model(self, name, namespace):
        result_obj = feature_selection_param_pb2.FeatureSelectionParam()
        model_manager.read_model(buffer_type="HeteroFeatureSelection{}.param".format(self.party_name),
                                 proto_buffer=result_obj,
                                 name=name,
                                 namespace=namespace)

        self.results = list(result_obj.results)
        left_col_obj = result_obj.final_left_cols
        self.cols = list(left_col_obj.original_cols)
        self.left_cols = dict(left_col_obj.left_cols)

    @staticmethod
    def select_cols(instance, left_cols, header):
        new_feature = []
        for col_idx, col_name in enumerate(header):
            is_left = left_cols.get(col_name)
            if not is_left:
                continue
            new_feature.append(instance.features[col_idx])
        new_feature = np.array(new_feature)
        instance.features = new_feature
        return instance

    def _reset_header(self):
        """
        The cols and left_cols record the index of header. Replace header based on the change
        between left_cols and cols.
        """
        new_header = []
        for col_name in self.header:
            is_left = self.left_cols.get(col_name)
            if is_left:
                new_header.append(col_name)
        self.header = new_header

    def _transfer_data(self, data_instances):

        if len(self.left_cols) == 0:
            raise ValueError("None left columns for feature selection. Please check if model has fit.")
        f = functools.partial(self.select_cols,
                              left_cols=self.left_cols,
                              header=self.header)

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

    def _renew_left_col_names(self):
        left_col_names = []
        for col_name, is_left in self.left_cols.items():
            if is_left:
                left_col_names.append(col_name)
        self.left_col_names = left_col_names

    def _init_cols(self, data_instances):
        header = get_header(data_instances)
        if self.cols == -1:
            self.cols = header

        self.left_col_names = self.cols.copy()
        self.header = header
        for col in self.cols:
            col_index = header.index(col)
            self.cols_dict[col] = col_index
