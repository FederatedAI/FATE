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
#
from federatedml.param.base_param import BaseParam

"""
Define how to transfer the cols

Parameters
----------
sum_cols : list of column index, default: None
    Specify which columns need to be sum. If column index is None, each of columns will be sum.

"""


class SecretSharingSumParam(BaseParam):
    def __init__(self, sum_cols=None):
        self.sum_cols = sum_cols
        if sum_cols is None:
            self.sum_cols = []

    def check(self):
        if isinstance(self.sum_cols, list):
            for idx in self.sum_cols:
                if not isinstance(idx, int):
                    raise ValueError(f"type mismatch, column_indexes with element {idx}(type is {type(idx)})")


