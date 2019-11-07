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

from federatedml.feature.feature_selection.unique_value_filter import UniqueValueFilter
from federatedml.feature.feature_selection import iv_value_select_filter
from federatedml.util import consts


def get_filter(filter_name, filter_param, role=consts.GUEST):
    if filter_name == consts.UNIQUE_VALUE:
        return UniqueValueFilter(filter_param)

    if filter_name == consts.IV_VALUE_THRES:
        if role == consts.GUEST:
            return iv_value_select_filter.Guest(filter_param)
        else:
            return iv_value_select_filter.Host(filter_param)
