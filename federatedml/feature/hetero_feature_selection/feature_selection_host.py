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

import numpy as np
import functools
from arch.api import eggroll
from arch.api import federation
from arch.api.proto.feature_engineer_result_pb2 import FeatureSelectResults, FeatureSelectFilterResult
from arch.api.utils import log_utils
from federatedml.feature import feature_selection
from federatedml.feature.feature_selection import FeatureSelection
from federatedml.feature.hetero_feature_binning.hetero_binning_host import HeteroFeatureHost
from federatedml.util import consts
from federatedml.util.fate_operator import get_features_shape
from federatedml.util.transfer_variable import HeteroFeatureSelectionTransferVariable

LOGGER = log_utils.getLogger()


class HeteroFeatureSelectionHost(object):
    def __init__(self, params):
        self.params = params
        self.cols = params.select_cols
        self.left_cols = None

        self.filter_method = params.filter_method
        self.feature_selection_method = FeatureSelection(self.params)

        self.bin_param = self.params.bin_param
        self.static_obj = None
        self.iv_attrs = None
        self.transfer_variable = HeteroFeatureSelectionTransferVariable()
        self.fit_iv = False
        self.receive_times = 0
        self.binning_obj = None
        self.results = []

    def fit(self, data_instances):
        if self.cols == -1:
            features_shape = get_features_shape(data_instances)
            if features_shape is None:
                raise RuntimeError('Cannot get feature shape, please check input data')
            self.cols = [i for i in range(features_shape)]
        self.left_cols = self.cols.copy()

        for method in self.filter_method:
            self.filter_one_method(data_instances, method)
            if len(self.left_cols) == 0:
                LOGGER.warning("After filter methods, none of feature left. Please check your filter parameters")
                break

    def transform(self, data_instances):
        new_data = self._transfer_data(data_instances)
        return new_data

    def fit_transform(self, data_instances):
        self.fit(data_instances)
        new_data = self.transform(data_instances)
        return new_data

    def fit_local(self, data_instances):
        feature_selection_obj = FeatureSelection(self.params)
        self.left_cols = feature_selection_obj.filter(data_instances)
        if self.cols == -1:
            self.cols = feature_selection_obj.select_cols

        self.left_cols = feature_selection_obj.filter(data_instances)
        self.results = feature_selection_obj.results

    def fit_local_transform(self, data_instances):
        self.fit_local(data_instances)
        new_data = self.transform(data_instances)
        return new_data

    def filter_one_method(self, data_instances, method):

        if method == consts.IV_VALUE_THRES:
            self._calculates_iv_attrs(data_instances)
            self._send_iv_threshold()
            self._received_result_cols(method=consts.IV_VALUE_THRES)
            LOGGER.info("Finish iv value threshold filter. Current left cols are: {}".format(self.left_cols))

        if method == consts.IV_PERCENTILE:
            self._calculates_iv_attrs(data_instances)
            self._received_result_cols(method=consts.IV_PERCENTILE)
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

    def _calculates_iv_attrs(self, data_instances):
        self.bin_param.cols = self.left_cols
        if self.binning_obj is None:
            self.binning_obj = HeteroFeatureHost(self.bin_param)
        else:
            self.binning_obj.reset(self.bin_param)
        self.binning_obj.fit(data_instances)
        LOGGER.info("Finish federated binning with guest.")

    def _received_result_cols(self, method):
        result_cols_id = self.transfer_variable.generate_transferid(self.transfer_variable.result_left_cols,
                                                                    self.receive_times)
        left_cols = federation.get(name=self.transfer_variable.result_left_cols.name,
                                   tag=result_cols_id,
                                   idx=0)
        self.receive_times += 1
        LOGGER.info("Received left columns from guest")
        new_left = []
        LOGGER.debug("Received left: {}, Original left: {}".format(left_cols, self.left_cols))
        for col in left_cols:
            new_left.append(self.left_cols[col])

        result_obj = FeatureSelectFilterResult(param_set={},
                                               original_cols=self.left_cols.copy(),
                                               left_cols=new_left.copy(),
                                               filter_name=method)
        self.results.append(result_obj)
        self.left_cols = new_left
        LOGGER.info("Received Left cols are {}".format(self.left_cols))

    def _send_iv_threshold(self):
        iv_thres_id = self.transfer_variable.generate_transferid(self.transfer_variable.host_iv_threshold)
        federation.remote(self.params.iv_param.value_threshold,
                          name=self.transfer_variable.host_iv_threshold.name,
                          tag=iv_thres_id,
                          role=consts.GUEST,
                          idx=0)
        LOGGER.info("Sent iv threshold to guest")

    def save_model(self):
        result_obj = FeatureSelectResults(results=self.results)
        serialize_str = result_obj.SerializeToString()
        meta_table = eggroll.parallelize([(1, serialize_str)],
                                         include_key=True,
                                         name=self.params.result_table,
                                         namespace=self.params.result_namespace,
                                         error_if_exist=False,
                                         persistent=True
                                         )
        LOGGER.info("Model saved, table: {}, namespace: {}".format(self.params.result_table,
                                                                   self.params.result_namespace))
        return meta_table

    def load_model(self, model_table, model_namespace):
        model = eggroll.table(model_table, model_namespace)
        model_local = model.collect()
        try:
            serialize_str = model_local.__next__()[1]
        except StopIteration:
            LOGGER.warning("Cannot load model from name_space: {}, model_table: {}".format(
                model_namespace, model_table
            ))
            return
        results = FeatureSelectResults()
        results.ParseFromString(serialize_str)
        self.results = results
        if len(self.results) == 0:
            self.left_cols = -1
        else:
            result_obj = self.results[-1]
            self.left_cols = result_obj.left_cols

    def _transfer_data(self, data_instances):
        if self.left_cols == -1:
            features_shape = get_features_shape(data_instances)
            if features_shape is None:
                raise RuntimeError('Cannot get feature shape, please check input data')
            self.left_cols = [i for i in range(features_shape)]

        f = functools.partial(self.select_cols,
                              left_cols=self.left_cols)

        new_data = data_instances.mapValues(f)
        return new_data

    @staticmethod
    def select_cols(instance, left_cols):
        new_feature = []
        for col in left_cols:
            new_feature.append(instance.features[col])
        new_feature = np.array(new_feature)
        instance.features = new_feature
        return instance
