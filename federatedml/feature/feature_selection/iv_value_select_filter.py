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
from federatedml.feature.feature_selection.selection_params import SelectionParams
from federatedml.framework.hetero.sync import selection_info_sync
from federatedml.param.feature_selection_param import IVValueSelectionParam
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class IVValueSelectFilter(BaseFilterMethod, metaclass=abc.ABCMeta):
    """
    filter the columns if all values in this feature is the same

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
        self.host_selection_inner_params = []
        self.sync_obj = selection_info_sync.Guest()

    def _parse_filter_param(self, filter_param):
        self.value_threshold = filter_param.value_threshold
        self.host_thresholds = filter_param.host_thresholds
        self.local_only = filter_param.local_only

    def set_host_party_ids(self, host_party_ids):
        if self.host_thresholds is None:
            self.host_thresholds = [self.value_threshold for _ in range(len(host_party_ids))]
        else:
            try:
                assert len(host_party_ids) == len(self.host_thresholds)
            except AssertionError:
                raise ValueError("Iv value filters param host_threshold set error."
                                 " The length should match host party numbers ")

    def fit(self, data_instances):
        self.selection_param = self.__unilateral_fit(self.binning_obj.binning_obj,
                                                     self.value_threshold,
                                                     self.selection_param)
        if not self.local_only:
            self.sync_obj.sync_select_cols()
            for host_id, host_threshold in enumerate(self.host_thresholds):
                self.__unilateral_fit(self.binning_obj.host_results[host_id],
                                      self.host_thresholds[host_id],
                                      self.host_selection_inner_params[host_id])

            self.sync_obj.sync_select_results(self.host_selection_inner_params)
        return self

    def __unilateral_fit(self, binning_model, threshold, selection_param):
        for col_name, col_results in binning_model.bin_results.all_cols_results.items():
            iv = col_results.iv
            if iv > threshold:
                selection_param.add_left_col_name(col_name)
                selection_param.add_feature_value(col_name, iv)
        return selection_param


class Host(IVValueSelectFilter):
    def __init__(self, filter_param: IVValueSelectionParam):
        super().__init__(filter_param)
        self.sync_obj = selection_info_sync.Host()

    def _parse_filter_param(self, filter_param):
        self.local_only = False

    def fit(self, data_instances):
        encoded_names = self.binning_obj.bin_inner_param.encode_col_name_list(self.selection_param.select_col_names)
        self.sync_obj.sync_select_cols(encoded_names)
        self.sync_obj.sync_select_results(self.selection_param)

