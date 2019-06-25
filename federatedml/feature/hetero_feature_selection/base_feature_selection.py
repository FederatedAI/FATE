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

from arch.api.proto import feature_selection_meta_pb2, feature_selection_param_pb2
from federatedml.model_base import ModelBase
from federatedml.param.param_feature_selection import FeatureSelectionParam
from federatedml.statistic.data_overview import get_header
from federatedml.util import abnormal_detection
from federatedml.util.transfer_variable import HeteroFeatureSelectionTransferVariable

MODEL_PARAM_NAME = 'FeatureSelectionParam'
MODEL_META_NAME = 'FeatureSelectionMeta'
MODEL_NAME = 'HeteroFeatureSelection'


class BaseHeteroFeatureSelection(ModelBase):
    def __init__(self):
        super(BaseHeteroFeatureSelection, self).__init__()
        self.transfer_variable = HeteroFeatureSelectionTransferVariable()
        self.cols = []
        self.left_col_names = []  # temp result
        self.left_cols = {}  # final result
        self.cols_dict = {}
        self.header = []
        self.party_name = 'Base'

        self.filter_meta_list = []
        self.filter_param_list = []

        # Possible previous model
        self.binning_model = None
        self.model_param = FeatureSelectionParam()

        # All possible meta
        self.unique_meta = None
        self.iv_value_meta = None
        self.iv_percentile_meta = None
        self.coe_meta = None
        self.outlier_meta = None

        # Use to save each model's result
        self.results = []

    def _init_model(self, params):
        self.model_param = params
        self.cols_index = params.select_cols
        self.filter_method = params.filter_method

    def _get_meta(self):
        meta_protobuf_obj = feature_selection_meta_pb2.FeatureSelectionMeta(filter_methods=self.filter_method,
                                                                            local_only=self.model_param.local_only,
                                                                            cols=self.cols,
                                                                            unique_meta=self.unique_meta,
                                                                            iv_value_meta=self.iv_value_meta,
                                                                            iv_percentile_meta=self.iv_percentile_meta,
                                                                            coe_meta=self.coe_meta,
                                                                            outlier_meta=self.outlier_meta)
        return meta_protobuf_obj

    def _get_param(self):
        left_col_obj = feature_selection_param_pb2.LeftCols(original_cols=self.cols,
                                                            left_cols=self.left_cols)

        result_obj = feature_selection_param_pb2.FeatureSelectionParam(results=self.results,
                                                                       final_left_cols=left_col_obj)
        return result_obj

    def save_data(self):
        return self.data_output

    def save_model(self):
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            MODEL_META_NAME: meta_obj,
            MODEL_PARAM_NAME: param_obj
        }
        return result

    def _load_model(self, model_dict):
        model_param = model_dict.get(MODEL_NAME).get(MODEL_PARAM_NAME)
        self.results = list(model_param.results)
        left_col_obj = model_param.final_left_cols
        self.cols = list(left_col_obj.original_cols)
        self.left_cols = dict(left_col_obj.left_cols)

    @staticmethod
    def select_cols(instance, left_cols, header):
        new_feature = []
        for col_idx, col_name in enumerate(header):
            is_left = left_cols.get(col_name)
            if is_left is None:
                continue
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

    def _renew_final_left_cols(self, new_left_cols):
        """
        As for all columns including those not specified in user params, record which columns left.
        """
        for col_name, is_left in new_left_cols.items():
            if not is_left:
                self.left_cols[col_name] = False

    def _init_cols(self, data_instances):
        header = get_header(data_instances)
        if self.cols_index == -1:
            self.cols = header
        else:
            cols = []
            for idx in self.cols_index:
                try:
                    idx = int(idx)
                except ValueError:
                    raise ValueError("In binning module, selected index: {} is not integer".format(idx))

                if idx >= len(header):
                    raise ValueError(
                        "In binning module, selected index: {} exceed length of data dimension".format(idx))
                cols.append(header[idx])
            self.cols = cols

        self.left_col_names = self.cols.copy()
        self.header = header
        for col in self.cols:
            col_index = header.index(col)
            self.cols_dict[col] = col_index
