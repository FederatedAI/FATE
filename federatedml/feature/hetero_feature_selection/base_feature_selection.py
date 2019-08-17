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
from google.protobuf import json_format

from arch.api.proto import feature_selection_meta_pb2, feature_selection_param_pb2
from arch.api.utils import log_utils
from federatedml.feature.hetero_feature_binning.hetero_binning_guest import HeteroFeatureBinningGuest
from federatedml.feature.hetero_feature_binning.hetero_binning_host import HeteroFeatureBinningHost
from federatedml.model_base import ModelBase
from federatedml.param.feature_selection_param import FeatureSelectionParam
from federatedml.statistic.data_overview import get_header
from federatedml.util import abnormal_detection
from federatedml.util import consts
from federatedml.util.transfer_variable.hetero_feature_selection_transfer_variable import \
    HeteroFeatureSelectionTransferVariable

LOGGER = log_utils.getLogger()

MODEL_PARAM_NAME = 'FeatureSelectionParam'
MODEL_META_NAME = 'FeatureSelectionMeta'
MODEL_NAME = 'HeteroFeatureSelection'


class BaseHeteroFeatureSelection(ModelBase):
    def __init__(self):
        super(BaseHeteroFeatureSelection, self).__init__()
        self.transfer_variable = HeteroFeatureSelectionTransferVariable()
        self.cols = []  # Current cols index to do selection
        # self.left_col_names = []
        self.left_cols = {}  # final result
        self.left_cols_index = []
        # self.cols_dict = {}
        self.header = []
        self.original_header = []
        self.schema = {}
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
        self.variance_coe_meta = None
        self.outlier_meta = None

        # Use to save each model's result
        self.results = []

    def _init_model(self, params):
        self.model_param = params
        self.cols_index = params.select_cols
        self.filter_methods = params.filter_methods
        self.local_only = params.local_only

    def _get_meta(self):
        cols = [str(i) for i in self.cols]
        meta_protobuf_obj = feature_selection_meta_pb2.FeatureSelectionMeta(filter_methods=self.filter_methods,
                                                                            local_only=self.model_param.local_only,
                                                                            cols=cols,
                                                                            unique_meta=self.unique_meta,
                                                                            iv_value_meta=self.iv_value_meta,
                                                                            iv_percentile_meta=self.iv_percentile_meta,
                                                                            variance_coe_meta=self.variance_coe_meta,
                                                                            outlier_meta=self.outlier_meta,
                                                                            need_run=self.need_run)
        return meta_protobuf_obj

    def _get_param(self):
        left_col_name_dict = {}

        LOGGER.debug("in _get_param, self.left_cols: {}, self.original_header: {}".format(
            self.left_cols, self.original_header
        ))

        for col_idx, is_left in self.left_cols.items():
            col_name = self.original_header[col_idx]
            left_col_name_dict[col_name] = is_left
        left_col_obj = feature_selection_param_pb2.LeftCols(original_cols=self.header,
                                                            left_cols=left_col_name_dict)

        result_obj = feature_selection_param_pb2.FeatureSelectionParam(results=self.results,
                                                                       final_left_cols=left_col_obj)
        json_result = json_format.MessageToJson(result_obj)
        LOGGER.debug("json_result: {}".format(json_result))
        return result_obj

    def save_data(self):
        return self.data_output

    def export_model(self):
        LOGGER.debug("Model output is : {}".format(self.model_output))
        if self.model_output is not None:
            LOGGER.debug("Model output is : {}".format(self.model_output))
            return self.model_output

        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            MODEL_META_NAME: meta_obj,
            MODEL_PARAM_NAME: param_obj
        }
        self.model_output = result
        return result

    def _load_model(self, model_dict):

        if 'model' in model_dict:
            # self._parse_need_run(model_dict, MODEL_META_NAME)
            LOGGER.debug("Feature selection need run: {}".format(self.need_run))
            if not self.need_run:
                return
            model_param = list(model_dict.get('model').values())[0].get(MODEL_PARAM_NAME)
            model_meta = list(model_dict.get('model').values())[0].get(MODEL_META_NAME)

            self.model_output = {
                MODEL_META_NAME: model_meta,
                MODEL_PARAM_NAME: model_param
            }
            LOGGER.debug("Model output set, model_output is :{}".format(self.model_output))
            self.results = list(model_param.results)
            left_col_obj = model_param.final_left_cols

            original_headers = list(left_col_obj.original_cols)
            self.header = original_headers
            self.cols = [int(i) for i in self.cols]
            left_col_name_dict = dict(left_col_obj.left_cols)
            LOGGER.debug("In load model, left_col_name_dict: {}, original_headers: {}".format(left_col_name_dict,
                                                                                              original_headers))
            for col_name, is_left in left_col_name_dict.items():
                col_idx = original_headers.index(col_name)
                self.left_cols[col_idx] = is_left
            LOGGER.debug("Self.left_cols: {}".format(self.left_cols))

        if 'isometric_model' in model_dict:

            LOGGER.debug("Has isometric_model, model_dict: {}".format(model_dict))
            if self.party_name == consts.GUEST:
                self.binning_model = HeteroFeatureBinningGuest()
            else:
                self.binning_model = HeteroFeatureBinningHost()

            new_model_dict = {'model': model_dict['isometric_model']}
            self.binning_model._load_model(new_model_dict)

    @staticmethod
    def select_cols(instance, left_cols, header):
        """
        Replace filtered columns to original data
        """
        new_feature = []

        for col_idx, col_name in enumerate(header):
            is_left = left_cols.get(col_idx)
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
        new_header = copy.deepcopy(self.header)
        for col_idx, col_name in enumerate(self.header):
            is_left = self.left_cols.get(col_idx)
            if is_left is None:
                continue
            if not is_left:
                new_header.remove(col_name)
        return new_header

    def _transfer_data(self, data_instances):

        if len(self.left_cols) == 0:
            raise ValueError("None left columns for feature selection. Please check if model has fit.")
        LOGGER.debug("In transfer_data, left_cols: {}, header: {}".format(self.left_cols, self.header))
        before_one_data = data_instances.first()
        f = functools.partial(self.select_cols,
                              left_cols=self.left_cols,
                              header=self.header)

        new_data = data_instances.mapValues(f)
        new_header = self._reset_header()
        # new_data.schema['header'] = new_header
        new_data = self.set_schema(new_data, new_header)

        one_data = new_data.first()[1]
        LOGGER.debug("In feature selection transform, Before transform: {}, length: {} After transform: {}, length: {}".format(
            before_one_data[1].features, len(before_one_data[1].features),
            one_data.features, len(one_data.features)))

        return new_data

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

    def _renew_final_left_cols(self, new_left_cols):
        """
        As for all columns including those not specified in user params, record which columns left.
        """
        left_col_list = []
        for col_idx, is_left in new_left_cols.items():
            if not is_left:
                self.left_cols[col_idx] = False
            elif is_left:
                left_col_list.append(col_idx)
                self.left_cols[col_idx] = True
        self.cols = left_col_list

    def _transform_init_cols(self, data_instances):
        self.schema = data_instances.schema
        header = get_header(data_instances)
        self.header = header

    def _init_cols(self, data_instances):
        self.schema = data_instances.schema
        header = get_header(data_instances)
        self.original_header = copy.deepcopy(header)
        LOGGER.debug("When init, original_header: {}".format(self.original_header))
        if self.cols_index == -1:
            self.cols = [i for i in range(len(header))]
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
                cols.append(idx)
            self.cols = cols

        # self.left_col_names = self.cols.copy()

        # Set all columns are left at the beginning.
        self.left_cols_index = [i for i in range(len(header))]
        for col_idx in self.cols:
            self.left_cols[col_idx] = True
        self.header = header

    def _generate_result_idx(self):
        self.left_col_names = []
        for col_idx, col_name in enumerate(self.header):
            if col_idx in self.left_cols_index:
                # self.left_col_names.append(self.header[col_idx])
                self.left_cols[col_name] = True
            else:
                self.left_cols[col_name] = False

    def set_schema(self, data_instance, header=None):
        if header is None:
            self.schema["header"] = self.header
        else:
            self.schema["header"] = header
        data_instance.schema = self.schema
        return data_instance
