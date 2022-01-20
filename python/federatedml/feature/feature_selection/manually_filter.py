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
from federatedml.param.feature_selection_param import ManuallyFilterParam
from federatedml.protobuf.generated import feature_selection_meta_pb2

from federatedml.util import LOGGER


class ManuallyFilter(BaseFilterMethod):
    def __init__(self, filter_param: ManuallyFilterParam):
        self.filter_out_indexes = None
        self.filter_out_names = None
        self.filter_param = None
        super().__init__(filter_param)

    def _parse_filter_param(self, filter_param):
        self.filter_param = filter_param

    def _transfer_params(self):
        header = self.selection_properties.header
        col_name_maps = self.selection_properties.col_name_maps
        if (self.filter_param.filter_out_indexes or self.filter_param.filter_out_names) is not None:
            self.filter_out_indexes = self.filter_param.filter_out_indexes
            self.filter_out_names = self.filter_param.filter_out_names
        elif (self.filter_param.left_col_indexes or self.filter_param.left_col_names) is not None:
            filter_out_set = set([i for i in range(len(header))])
            if self.filter_param.left_col_indexes is not None:
                filter_out_set = filter_out_set.difference(self.filter_param.left_col_indexes)
            if self.filter_param.left_col_names is not None:
                left_idx = [col_name_maps.get(name) for name in self.filter_param.left_col_names]
                filter_out_set = filter_out_set.difference(left_idx)
            self.filter_out_indexes = list(filter_out_set)

        if self.filter_out_indexes is None:
            self.filter_out_indexes = []

        if self.filter_out_names is None:
            self.filter_out_names = []

    def fit(self, data_instances, suffix):
        self._transfer_params()
        all_filter_out_names = []
        for col_idx, col_name in zip(self.selection_properties.select_col_indexes,
                                     self.selection_properties.select_col_names):
            # LOGGER.debug("Col_idx: {}, col_names: {}, filter_out_indexes: {}, filter_out_names: {}".format(
            #    col_idx, col_name, self.filter_out_indexes, self.filter_out_names
            # ))
            if col_idx not in self.filter_out_indexes and col_name not in self.filter_out_names:
                self.selection_properties.add_left_col_name(col_name)
            else:
                all_filter_out_names.append(col_name)
        self._keep_one_feature()
        self.filter_out_names = all_filter_out_names
        # LOGGER.debug(f"filter out names are: {self.filter_out_names}")
        return self

    # def get_meta_obj(self, meta_dicts):
    #     result = feature_selection_meta_pb2.ManuallyFilterMeta(filter_out_names=self.filter_out_names)
    #     meta_dicts['manually_meta'] = result
    #     return meta_dicts

    def get_meta_obj(self):
        result = feature_selection_meta_pb2.FilterMeta()
        return result
