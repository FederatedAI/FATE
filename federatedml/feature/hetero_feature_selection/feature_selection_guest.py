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

from arch.api import federation
from arch.api.model_manager import manager as model_manager
from arch.api.proto import feature_selection_meta_pb2, feature_selection_param_pb2
from arch.api.proto.feature_binning_meta_pb2 import FeatureBinningMeta
from arch.api.utils import log_utils
from federatedml.feature import feature_selection
from federatedml.feature.feature_selection import FeatureSelection
from federatedml.feature.hetero_feature_binning.hetero_binning_guest import HeteroFeatureBinningGuest
from federatedml.param.param import IVSelectionParam
from federatedml.util import consts
from federatedml.util.fate_operator import get_features_shape
from federatedml.util.transfer_variable import HeteroFeatureSelectionTransferVariable

LOGGER = log_utils.getLogger()


class HeteroFeatureSelectionGuest(object):
    def __init__(self, params):
        self.params = params
        self.cols = params.select_cols
        self.left_cols = None
        self.host_left_cols = None
        self.filter_method = params.filter_method
        self.local_only = params.local_only
        self.guest_iv_attrs = None
        self.host_iv_attrs = None
        self.bin_param = self.params.bin_param
        self.static_obj = None
        self.transfer_variable = HeteroFeatureSelectionTransferVariable()
        self.send_times = 0
        self.binning_model = None
        self.results = []
        self.header = []
        self.flowid = ''

    def fit(self, data_instances):
        self.header = data_instances.schema.get('header')  # ['x1', 'x2', 'x3' ... ]

        self._parse_cols(data_instances)
        self.left_cols = self.cols.copy()

        for method in self.filter_method:
            self.filter_one_method(data_instances, method)
            if len(self.left_cols) == 0:
                LOGGER.warning("After filter methods, none of feature left. Please check your filter parameters")
                break

    def fit_local(self, data_instances):
        self.header = data_instances.schema.get('header')  # ['x1', 'x2', 'x3' ... ]

        feature_selection_obj = FeatureSelection(self.params)
        self.left_cols = feature_selection_obj.filter(data_instances)
        if self.cols == -1:
            self.cols = feature_selection_obj.select_cols

        self.left_cols = feature_selection_obj.filter(data_instances)
        self.results = feature_selection_obj.results

    def fit_local_transform(self, data_instances):
        self._parse_cols(data_instances)
        self.header = data_instances.schema.get('header')  # ['x1', 'x2', 'x3' ... ]
        self.fit_local(data_instances)
        new_data = self.transform(data_instances)
        new_data.schema['header'] = self.header

        return new_data

    def transform(self, data_instances):
        self._parse_cols(data_instances)
        self.header = data_instances.schema.get('header')  # ['x1', 'x2', 'x3' ... ]
        new_data = self._transfer_data(data_instances)
        new_data.schema['header'] = self.header

        return new_data

    def fit_transform(self, data_instances):
        self.header = data_instances.schema.get('header')  # ['x1', 'x2', 'x3' ... ]
        self.fit(data_instances)
        new_data = self.transform(data_instances)
        new_data.schema['header'] = self.header
        return new_data

    def filter_one_method(self, data_instances, method):

        if method == consts.IV_VALUE_THRES:
            self._calculates_iv_attrs(data_instances, flowid_postfix='iv_value')
            iv_param = self.params.iv_param
            iv_filter = feature_selection.IVValueSelectFilter(iv_param, self.left_cols, self.guest_iv_attrs)
            new_left_cols = iv_filter.filter()

            self.results.append(iv_filter.to_result())

            # Renew current left cols and iv_attrs
            new_iv_list = self._renew_iv_attrs(new_left_cols, self.left_cols, self.guest_iv_attrs)
            self.guest_iv_attrs = new_iv_list
            self.left_cols = new_left_cols

            if not self.local_only:
                self._filter_host_iv_value()
            LOGGER.info("Finish iv value threshold filter. Current left cols are: {}".format(self.left_cols))

        if method == consts.IV_PERCENTILE:
            self._calculates_iv_attrs(data_instances, flowid_postfix='iv_percentile')
            iv_param = self.params.iv_param
            iv_filter = feature_selection.IVPercentileFilter(iv_param)
            iv_filter.add_attrs(self.guest_iv_attrs, self.left_cols)
            if not self.local_only:
                # self.host_left_cols = [x for x in range(len(self.host_iv_attrs))]
                iv_filter.add_attrs(self.host_iv_attrs, self.host_left_cols)
            left_cols = iv_filter.filter_multiple_parties()
            # LOGGER.debug("In iv percentile function, left cols are: {}".format(left_cols))
            new_left_cols = left_cols[0]
            self.results.append(iv_filter.to_result())

            # Renew current left cols and iv_attrs
            new_iv_list = self._renew_iv_attrs(new_left_cols, self.left_cols, self.guest_iv_attrs)
            self.guest_iv_attrs = new_iv_list
            self.left_cols = new_left_cols

            # If host has participated, send result to host
            if len(left_cols) > 1:
                new_host_left_cols = left_cols[1]
                new_host_iv_list = self._renew_iv_attrs(new_host_left_cols, self.host_left_cols, self.host_iv_attrs)
                self.host_iv_attrs = new_host_iv_list
                self.host_left_cols = new_host_left_cols
                self._send_host_result_cols()
            LOGGER.info("Finish iv percentile filter. Current left cols are: {}".format(self.left_cols))

        if method == consts.COEFFICIENT_OF_VARIATION_VALUE_THRES:
            coe_param = self.params.coe_param
            coe_filter = feature_selection.CoeffOfVarValueFilter(coe_param, self.left_cols, self.static_obj)
            self.left_cols = coe_filter.filter(data_instances)
            self.static_obj = coe_filter.statics_obj
            self.results.append(coe_filter.to_result())

            LOGGER.info("Finish coeffiecient_of_variation value threshold filter. Current left cols are: {}".format(
                self.left_cols))

        if method == consts.UNIQUE_VALUE:
            unique_param = self.params.unique_param
            unique_filter = feature_selection.UniqueValueFilter(unique_param, self.left_cols, self.static_obj)
            self.left_cols = unique_filter.filter(data_instances)
            self.static_obj = unique_filter.statics_obj
            self.results.append(unique_filter.to_result())

            LOGGER.info("Finish unique value filter. Current left cols are: {}".format(
                self.left_cols))

        if method == consts.OUTLIER_COLS:
            outlier_param = self.params.outlier_param
            outlier_filter = feature_selection.OutlierFilter(outlier_param, self.left_cols)
            self.left_cols = outlier_filter.filter(data_instances)
            self.results.append(outlier_filter.to_result())
            LOGGER.info("Finish outlier cols filter. Current left cols are: {}".format(
                self.left_cols))

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

        # LOGGER.debug("self.left_cols: {}".format(self.left_cols))
        f = functools.partial(self.select_cols,
                              left_cols=self.left_cols)

        new_data = data_instances.mapValues(f)
        self._reset_header()
        return new_data

    @staticmethod
    def select_cols(instance, left_cols):
        new_feature = []
        for col in left_cols:
            new_feature.append(instance.features[col])
        new_feature = np.array(new_feature)
        instance.features = new_feature
        return instance

    def _calculates_iv_attrs(self, data_instances, flowid_postfix=''):
        if self.local_only and self.guest_iv_attrs is not None:
            return

        bin_flow_id = self.flowid + flowid_postfix
        self.bin_param.cols = self.left_cols
        if self.binning_model is None:
            self.binning_model = HeteroFeatureBinningGuest(self.bin_param)
            self.binning_model.set_flowid(bin_flow_id)
        else:
            self.binning_model.reset(self.bin_param, flowid=bin_flow_id)

        if self.local_only:
            if self.guest_iv_attrs is None:
                self.guest_iv_attrs = self.binning_model.fit_local(data_instances=data_instances)
        else:
            iv_attrs = self.binning_model.fit(data_instances)
            self.guest_iv_attrs = iv_attrs.get('local')
            self.host_iv_attrs = iv_attrs.get('remote')
            self.host_left_cols = [i for i in range(len(self.host_iv_attrs))]
            LOGGER.debug("Host left cols: {}".format(self.host_left_cols))
        LOGGER.info("Finish federated binning with host.")

    def _send_host_result_cols(self):
        result_cols_id = self.transfer_variable.generate_transferid(self.transfer_variable.result_left_cols,
                                                                    self.send_times)
        federation.remote(self.host_left_cols,
                          name=self.transfer_variable.result_left_cols.name,
                          tag=result_cols_id,
                          role=consts.HOST,
                          idx=0)
        self.send_times += 1
        LOGGER.info("Sent result cols from guest to host, result cols are: {}".format(self.host_left_cols))

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

    def _filter_host_iv_value(self):
        host_iv_thres_id = self.transfer_variable.generate_transferid(self.transfer_variable.host_iv_threshold)
        host_iv_thres = federation.get(name=self.transfer_variable.host_iv_threshold.name,
                                       tag=host_iv_thres_id,
                                       idx=0)
        LOGGER.info("Received iv threshold from host, threshold is :{}".format(host_iv_thres))
        iv_param = IVSelectionParam(value_threshold=host_iv_thres)
        host_filter = feature_selection.IVValueSelectFilter(iv_param, self.host_left_cols, self.host_iv_attrs)
        new_host_left_cols = host_filter.filter()

        # Renew current host left cols and host iv_attrs
        self.host_iv_attrs = self._renew_iv_attrs(new_host_left_cols, self.host_left_cols, self.host_iv_attrs)
        self.host_left_cols = new_host_left_cols

        self._send_host_result_cols()

    def _renew_iv_attrs(self, new_left_cols, pre_left_cols, iv_attrs):
        new_iv_list = []
        for left_col in new_left_cols:
            idx = pre_left_cols.index(left_col)
            new_iv_list.append(iv_attrs[idx])
        return new_iv_list

    def _parse_cols(self, data_instances):
        if self.cols == -1:
            features_shape = get_features_shape(data_instances)
            if features_shape is None:
                raise RuntimeError('Cannot get feature shape, please check input data')
            self.cols = [i for i in range(features_shape)]

    def set_flowid(self, flowid="samole"):
        self.flowid = flowid
        self.transfer_variable.set_flowid(self.flowid)
