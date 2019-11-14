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
from federatedml.feature.feature_selection.selection_properties import SelectionProperties
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class Guest(object):
    # noinspection PyAttributeOutsideInit
    def register_selection_trans_vars(self, transfer_variable):
        self._host_select_cols_transfer = transfer_variable.host_select_cols
        self._result_left_cols_transfer = transfer_variable.result_left_cols

    def sync_select_cols(self):
        host_select_col_names = self._host_select_cols_transfer.get(idx=-1)
        host_selection_params = []
        for host_id, select_names in enumerate(host_select_col_names):
            host_selection_properties = SelectionProperties()
            host_selection_properties.set_header(select_names)
            host_selection_properties.add_select_col_names(select_names)
            host_selection_params.append(host_selection_properties)
        return host_selection_params

    def sync_select_results(self, host_selection_inner_params):
        for host_id, host_select_results in enumerate(host_selection_inner_params):
            self._result_left_cols_transfer.remote(host_select_results.left_col_names,
                                                   role=consts.HOST,
                                                   idx=host_id)


class Host(object):
    # noinspection PyAttributeOutsideInit
    def register_selection_trans_vars(self, transfer_variable):
        self._host_select_cols_transfer = transfer_variable.host_select_cols
        self._result_left_cols_transfer = transfer_variable.result_left_cols

    def sync_select_cols(self, encoded_names):
        self._host_select_cols_transfer.host_select_cols.remote(encoded_names,
                                                                role=consts.GUEST,
                                                                idx=0)

    def sync_select_results(self, selection_param, decode_func=None):
        left_cols_names = self._result_left_cols_transfer.get(idx=0)
        for col_name in left_cols_names:
            if decode_func is not None:
                col_name = decode_func(col_name)
            selection_param.add_left_col_name(col_name)
