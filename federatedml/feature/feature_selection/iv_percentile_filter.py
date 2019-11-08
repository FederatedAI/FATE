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

from federatedml.feature.feature_selection.filter_base import BaseFilterMethod
from federatedml.param.feature_selection_param import IVPercentileSelectionParam
from federatedml.framework.hetero.sync import selection_info_sync
from federatedml.feature.hetero_feature_binning.base_feature_binning import BaseHeteroFeatureBinning
from federatedml.util import consts
import abc


class IVPercentileFilter(BaseFilterMethod, metaclass=abc.ABCMeta):
    def __init__(self, filter_param):
        super().__init__(filter_param)
        self.transfer_variable = None
        self.binning_obj: BaseHeteroFeatureBinning = None
        self.local_only = False
        self.sync_obj = None

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable
        self.sync_obj.register_selection_trans_vars(transfer_variable)

    def _parse_filter_param(self, filter_param):
        self.percentile_threshold = filter_param.percentile_threshold
        self.local_only = filter_param.local_only

    def set_binning_obj(self, binning_model):
        if binning_model is None:
            raise ValueError("To use iv filter, binning module should be called and setup in 'isomatric_model'"
                             " input for feature selection.")
        self.binning_obj = binning_model


class Guest(IVPercentileFilter):
    def __init__(self, filter_param: IVPercentileSelectionParam):
        super().__init__(filter_param)
        self.host_thresholds = None
        self.host_selection_inner_params = []
        self.sync_obj = selection_info_sync.Guest()

    def fit(self, data_instances):
        pass

    def get_value_threshold(self):
        total_values = []
        for col_name, iv in self.binning_obj.binning_obj.bin_results.all_cols_results.items():
            if col_name in self.selection_param.select_col_names:
                total_values.append(iv)


        if not self.local_only:
            for host_id, host_binning_obj in self.binning_obj.host_results.items():
                # TODO
                pass




class Host(IVPercentileFilter):
    def __init__(self, filter_param: IVPercentileSelectionParam):
        super().__init__(filter_param)
        self.sync_obj = selection_info_sync.Host()