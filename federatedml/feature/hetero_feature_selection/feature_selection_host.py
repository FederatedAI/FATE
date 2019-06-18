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

from arch.api import federation
from arch.api.proto import feature_selection_param_pb2
from arch.api.utils import log_utils
from federatedml.feature import feature_selection
from federatedml.feature.hetero_feature_selection.base_feature_selection import BaseHeteroFeatureSelection
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroFeatureSelectionHost(BaseHeteroFeatureSelection):
    def __init__(self, params):
        super(HeteroFeatureSelectionHost, self).__init__(params)

        self.static_obj = None
        self.iv_attrs = None
        self.fit_iv = False
        self.binning_obj = None
        self.results = []
        self.header = []
        self.flowid = ''
        self.party_name = consts.HOST

    def fit_transform(self, data_instances):
        self._abnormal_detection(data_instances)
        self._init_cols(data_instances)
        LOGGER.debug("host data count: {}, host header: {}".format(data_instances.count(), self.header))
        for method in self.filter_method:
            self.filter_one_method(data_instances, method)
            self._renew_left_col_names()

        new_data = self._transfer_data(data_instances)
        self._reset_header()
        new_data.schema['header'] = self.header
        return data_instances

    def transform(self, data_instances):
        self._abnormal_detection(data_instances)

        self._init_cols(data_instances)
        LOGGER.info("[Result][FeatureSelection][Host]In transform, Self left cols are: {}".format(
            self.left_cols
        ))
        new_data = self._transfer_data(data_instances)
        self._reset_header()
        new_data.schema['header'] = self.header

        return new_data

    def fit(self, data_instances):

        self._abnormal_detection(data_instances)

        self._init_cols(data_instances)

        for method in self.filter_method:
            self.filter_one_method(data_instances, method)
            self._renew_left_col_names()

        data_instances.schema['header'] = self.header
        return data_instances

    def filter_one_method(self, data_instances, method):

        if method == consts.IV_VALUE_THRES:
            self._send_select_cols(consts.IV_VALUE_THRES)
            self._received_result_cols(filter_name=consts.IV_VALUE_THRES)
            LOGGER.info(
                "[Result][FeatureSelection][Host]Finish iv value threshold filter. Current left cols are: {}".format(
                    self.left_cols))

        if method == consts.IV_PERCENTILE:
            self._send_select_cols(consts.IV_PERCENTILE)
            self._received_result_cols(filter_name=consts.IV_PERCENTILE)
            LOGGER.info("[Result][FeatureSelection][Host]Finish iv percentile filter. Current left cols are: {}".format(
                self.left_cols))

        if method == consts.COEFFICIENT_OF_VARIATION_VALUE_THRES:
            coe_param = self.params.coe_param
            coe_filter = feature_selection.CoeffOfVarValueFilter(coe_param, self.left_col_names, self.static_obj)
            self.left_cols = coe_filter.fit(data_instances)
            self.static_obj = coe_filter.statics_obj
            self.coe_meta = coe_filter.get_meta_obj()
            self.results.append(coe_filter.get_param_obj())
            self._renew_left_col_names()
            coe_filter.display_feature_result(self.party_name)
            LOGGER.info(
                "[Result][FeatureSelection][Host]Finish coeffiecient_of_variation value threshold filter."
                " Current left cols are: {}".format(
                    self.left_cols))

        if method == consts.UNIQUE_VALUE:
            unique_param = self.params.unique_param
            unique_filter = feature_selection.UniqueValueFilter(unique_param, self.left_col_names, self.static_obj)
            self.left_cols = unique_filter.fit(data_instances)
            self.static_obj = unique_filter.statics_obj
            self.unique_meta = unique_filter.get_meta_obj()
            self.results.append(unique_filter.get_param_obj())
            self._renew_left_col_names()
            unique_filter.display_feature_result(self.party_name)
            LOGGER.info("[Result][FeatureSelection][Host]Finish unique value filter. Current left cols are: {}".format(
                self.left_cols))

        if method == consts.OUTLIER_COLS:
            outlier_param = self.params.outlier_param
            outlier_filter = feature_selection.OutlierFilter(outlier_param, self.left_col_names)
            self.left_cols = outlier_filter.fit(data_instances)
            self.outlier_meta = outlier_filter.get_meta_obj()
            self.results.append(outlier_filter.get_param_obj())
            self._renew_left_col_names()
            outlier_filter.display_feature_result(self.party_name)
            LOGGER.info("[Result][FeatureSelection][Host]Finish outlier cols filter. Current left cols are: {}".format(
                self.left_cols))

    def _received_result_cols(self, filter_name):
        result_cols_id = self.transfer_variable.generate_transferid(self.transfer_variable.result_left_cols,
                                                                    filter_name)
        left_cols = federation.get(name=self.transfer_variable.result_left_cols.name,
                                   tag=result_cols_id,
                                   idx=0)
        LOGGER.info("Received left columns from guest")
        self.left_cols = left_cols
        self._renew_left_col_names()

        host_cols = list(left_cols.keys())
        left_col_obj = feature_selection_param_pb2.LeftCols(original_cols=host_cols,
                                                            left_cols=self.left_cols)

        result_obj = feature_selection_param_pb2.FeatureSelectionFilterParam(feature_values={},
                                                                             left_cols=left_col_obj,
                                                                             filter_name=filter_name)
        self.results.append(result_obj)
        LOGGER.info("Received Left cols are {}".format(self.left_cols))

    def _send_select_cols(self, filter_name):
        host_select_cols_id = self.transfer_variable.generate_transferid(self.transfer_variable.host_select_cols,
                                                                         filter_name)
        federation.remote(self.left_col_names,
                          name=self.transfer_variable.host_select_cols.name,
                          tag=host_select_cols_id,
                          role=consts.GUEST,
                          idx=0)
        LOGGER.info("Sent select cols to guest")
