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

import abc

from arch.api.utils import log_utils
from federatedml.feature.feature_selection.filter_base import BaseFilterMethod
from federatedml.framework.hetero.sync import selection_info_sync
from federatedml.param.feature_selection_param import IVValueSelectionParam
from federatedml.protobuf.generated import feature_selection_meta_pb2
from federatedml.feature.feature_selection.selection_properties import SelectionProperties

LOGGER = log_utils.getLogger()


def fit_iv_values(binning_model, threshold, selection_param: SelectionProperties):
    alternative_col_name = None
    for col_name, col_results in binning_model.bin_results.all_cols_results.items():
        if col_name not in selection_param.select_col_names:
            continue
        alternative_col_name = col_name
        iv = col_results.iv
        if iv > threshold:
            selection_param.add_left_col_name(col_name)
        selection_param.add_feature_value(col_name, iv)
    if len(selection_param.all_left_col_names) == 0:
        assert alternative_col_name is not None
        selection_param.add_left_col_name(alternative_col_name)
    return selection_param


class IVValueSelectFilter(BaseFilterMethod, metaclass=abc.ABCMeta):
    """
    filter the columns if iv value is less than a threshold
    """

    def __init__(self, filter_param: IVValueSelectionParam):
        super().__init__(filter_param)
        self.binning_obj = None
        self.local_only = False
        self.transfer_variable = None
        self.sync_obj = None

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable
        self.sync_obj.register_selection_trans_vars(transfer_variable)

    def set_binning_obj(self, binning_model):
        if binning_model is None:
            raise ValueError("To use iv filter, binning module should be called and setup in 'isomatric_model'"
                             " input for feature selection.")
        self.binning_obj = binning_model


class Guest(IVValueSelectFilter):
    def __init__(self, filter_param: IVValueSelectionParam):
        super().__init__(filter_param)
        self.host_thresholds = None
        self.host_selection_properties = []
        self.sync_obj = selection_info_sync.Guest()

    def _parse_filter_param(self, filter_param):
        self.value_threshold = filter_param.value_threshold
        self.host_thresholds = filter_param.host_thresholds
        self.local_only = filter_param.local_only

    def fit(self, data_instances, suffix):
        self.selection_properties = fit_iv_values(self.binning_obj.binning_obj,
                                                  self.value_threshold,
                                                  self.selection_properties)
        if not self.local_only:
            self.host_selection_properties = self.sync_obj.sync_select_cols(suffix=suffix)
            for host_id, host_properties in enumerate(self.host_selection_properties):
                if self.host_thresholds is None:
                    threshold = self.value_threshold
                else:
                    threshold = self.host_thresholds[host_id]
                LOGGER.debug("host_properties.header: {}, host_bin_results: {}".format(
                    host_properties.header, self.binning_obj.host_results[host_id].bin_results.all_cols_results))

                fit_iv_values(self.binning_obj.host_results[host_id],
                              threshold,
                              host_properties)
                LOGGER.debug("In iv_value fit, host_properties.left_col_indexes: {}, last_left_col_indexes: {}".format(
                    host_properties.left_col_indexes, host_properties.last_left_col_indexes
                ))

            self.sync_obj.sync_select_results(self.host_selection_properties, suffix=suffix)
        return self

    def get_meta_obj(self, meta_dicts):
        result = feature_selection_meta_pb2.IVValueSelectionMeta(value_threshold=self.value_threshold,
                                                                 local_only=self.local_only)
        meta_dicts['iv_value_meta'] = result
        return meta_dicts


class Host(IVValueSelectFilter):
    def __init__(self, filter_param: IVValueSelectionParam):
        super().__init__(filter_param)
        self.sync_obj = selection_info_sync.Host()

    def _parse_filter_param(self, filter_param):
        self.local_only = False

    def fit(self, data_instances, suffix):
        encoded_names = self.binning_obj.bin_inner_param.encode_col_name_list(
            self.selection_properties.select_col_names)
        LOGGER.debug("selection_properties.select_col_names: {}, encoded_names: {}".format(
            self.selection_properties.select_col_names, encoded_names
        ))

        self.sync_obj.sync_select_cols(encoded_names, suffix=suffix)
        self.sync_obj.sync_select_results(self.selection_properties,
                                          decode_func=self.binning_obj.bin_inner_param.decode_col_name,
                                          suffix=suffix)
        LOGGER.debug("In fit selected result, left_col_names: {}".format(self.selection_properties.left_col_names))
        return self

    def get_meta_obj(self, meta_dicts):
        result = feature_selection_meta_pb2.IVValueSelectionMeta(local_only=self.local_only)
        meta_dicts['iv_value_meta'] = result
        return meta_dicts
