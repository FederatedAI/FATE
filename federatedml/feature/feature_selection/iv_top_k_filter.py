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

import operator

from arch.api.utils import log_utils
from federatedml.feature.feature_selection.iv_percentile_filter import IVPercentileFilter
from federatedml.framework.hetero.sync import selection_info_sync
from federatedml.param.feature_selection_param import IVPercentileSelectionParam
from federatedml.protobuf.generated import feature_selection_meta_pb2

LOGGER = log_utils.getLogger()


class Guest(IVPercentileFilter):
    def __init__(self, filter_param: IVPercentileSelectionParam):
        super().__init__(filter_param)
        self.host_selection_properties = []
        self.sync_obj = selection_info_sync.Guest()

    def _parse_filter_param(self, filter_param):
        self.k = filter_param.k
        self.local_only = filter_param.local_only

    def fit(self, data_instances, suffix):
        if not self.local_only:
            self.host_selection_properties = self.sync_obj.sync_select_cols(suffix=suffix)

        sorted_iv = self.get_sorted_iv()
        substitute_col_guest = None  # Used if none has been left
        substitute_col_host = {}
        for idx, (col_info, iv) in enumerate(sorted_iv):
            if col_info[0] == 'guest':
                substitute_col_guest = col_info[1]
                if idx < self.k:
                    self.selection_properties.add_left_col_name(col_info[1])
                self.selection_properties.add_feature_value(col_info[1], iv)
            else:
                host_id = col_info[0]
                substitute_col_host[host_id] = col_info[1]
                if idx < self.k:
                    self.host_selection_properties[host_id].add_left_col_name(col_info[1])
                self.host_selection_properties[host_id].add_feature_value(col_info[1], iv)

        if len(self.selection_properties.all_left_col_names) == 0:
            self.selection_properties.add_left_col_name(substitute_col_guest)

        for host_id, host_col_name in substitute_col_host.items():
            if len(self.host_selection_properties[host_id].all_left_col_names) == 0:
                self.host_selection_properties[host_id].add_left_col_name(host_col_name)

        if not self.local_only:
            self.sync_obj.sync_select_results(self.host_selection_properties, suffix=suffix)
        return self

    def get_sorted_iv(self):
        all_iv_map = {}
        for col_name, col_results in self.binning_obj.binning_obj.bin_results.all_cols_results.items():
            if col_name in self.selection_properties.select_col_names:
                all_iv_map[("guest", col_name)] = col_results.iv

        if not self.local_only:
            for host_id, host_binning_obj in enumerate(self.binning_obj.host_results):
                host_select_param = self.host_selection_properties[host_id]
                for col_name, col_results in host_binning_obj.bin_results.all_cols_results.items():
                    if col_name in host_select_param.select_col_names:
                        all_iv_map[(host_id, col_name)] = col_results.iv

        result = sorted(all_iv_map.items(), key=operator.itemgetter(1), reverse=True)
        return result

    def get_meta_obj(self, meta_dicts):
        result = feature_selection_meta_pb2.IVTopKSelectionMeta(k=self.k,
                                                                local_only=self.local_only)
        meta_dicts['iv_top_k_meta'] = result
        return meta_dicts


class Host(IVPercentileFilter):
    def __init__(self, filter_param: IVPercentileSelectionParam):
        super().__init__(filter_param)
        self.sync_obj = selection_info_sync.Host()

    def _parse_filter_param(self, filter_param):
        self.local_only = False

    def fit(self, data_instances, suffix):
        encoded_names = self.binning_obj.bin_inner_param.encode_col_name_list(
            self.selection_properties.select_col_names)
        self.sync_obj.sync_select_cols(encoded_names, suffix=suffix)
        self.sync_obj.sync_select_results(self.selection_properties,
                                          decode_func=self.binning_obj.bin_inner_param.decode_col_name,
                                          suffix=suffix)
        return self

    def get_meta_obj(self, meta_dicts):
        result = feature_selection_meta_pb2.IVTopKSelectionMeta(local_only=self.local_only)
        meta_dicts['iv_top_k_meta'] = result
        return meta_dicts
