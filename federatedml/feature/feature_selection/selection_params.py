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

from arch.api.utils import log_utils
from federatedml.protobuf.generated import feature_selection_meta_pb2, feature_selection_param_pb2

LOGGER = log_utils.getLogger()


class SelectionParams(object):
    def __init__(self):
        self.header = []
        self.col_name_maps = {}
        self.last_left_col_indexes = []
        self.select_col_indexes = []
        self.select_col_names = []
        self.left_col_indexes = []
        self.left_col_names = []
        self.feature_values = {}

    def set_header(self, header):
        self.header = header
        for idx, col_name in enumerate(self.header):
            self.col_name_maps[col_name] = idx

    def set_last_left_col_indexes(self, left_cols):
        self.last_left_col_indexes = left_cols

    def add_select_col_indexes(self, select_col_indexes):
        for idx in select_col_indexes:
            if idx >= len(self.header):
                LOGGER.warning("Adding a index that out of header's bound")
                continue
            if idx not in self.last_left_col_indexes:
                continue

            if idx not in self.select_col_indexes:
                self.select_col_indexes.append(idx)
                self.select_col_names.append(self.header[idx])

    def add_select_col_names(self, select_col_names):
        for col_name in select_col_names:
            idx = self.col_name_maps.get(col_name)
            if idx is None:
                LOGGER.warning("Adding a col_name that is not exist in header")
                continue
            if idx not in self.last_left_col_indexes:
                continue
            if idx not in self.select_col_indexes:
                self.select_col_indexes.append(idx)
                self.select_col_names.append(self.header[idx])

    def add_left_col_name(self, left_col_name):
        idx = self.col_name_maps.get(left_col_name)
        if idx is None:
            LOGGER.warning("Adding a col_name that is not exist in header")
        if idx not in self.select_col_indexes:
            self.select_col_indexes.append(idx)
            self.select_col_names.append(self.header[idx])

    def add_feature_value(self, col_name, feature_value):
        self.feature_values[col_name] = feature_value

    @property
    def all_left_col_indexes(self):
        result = []
        for idx in self.last_left_col_indexes:
            if idx not in self.select_col_indexes:
                result.append(idx)
            elif idx in self.left_col_indexes:
                result.append(idx)
        return result

    @property
    def all_left_col_names(self):
        left_indexes = self.all_left_col_indexes
        return [self.header[x] for x in left_indexes]

    @property
    def last_left_col_names(self):
        return [self.header[x] for x in self.last_left_col_indexes]


class FeatureSelectionFilterParam(object):
    def __init__(self):
        self.feature_values = {}
        self.host_feature_values = []
        self.left_cols = None
        self.host_left_cols = None
        self.filter_name = ''


class CompletedSelectionResults(object):
    def __init__(self):
        self.header = []
        self.col_name_maps = {}
        self.filter_results = []

    def set_header(self, header):
        self.header = header
        for idx, col_name in enumerate(self.header):
            self.col_name_maps[col_name] = idx

    def add_filter_results(self, filter_name, select_param: SelectionParams, host_select_params: list=None):
        if host_select_params is None:
            host_select_params = []

        host_feature_values = []
        host_left_cols = []
        for host_result in host_select_params:
            feature_value_pb = feature_selection_param_pb2.FeatureValue(feature_values=host_result.feature_values)
            host_feature_values.append(feature_value_pb)
            left_col_pb = feature_selection_param_pb2.LeftCols(original_cols=host_result.last_left_col_names,
                                                               left_cols=host_result.all_left_col_names)
            host_left_cols.append(left_col_pb)

        this_filter_result = feature_selection_param_pb2. \
            FeatureSelectionFilterParam(feature_values=select_param.feature_values,
                                        host_feature_values=host_feature_values,
                                        left_cols=select_param.all_left_col_names,
                                        host_left_cols=host_left_cols,
                                        filter_name=filter_name)
        self.filter_results.append(this_filter_result)






