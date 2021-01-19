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
from pipeline.param.base_param import BaseParam

"""
Define how to transfer the cols

Parameters
----------
sum_cols : list of column index, default: None
    Specify which columns need to be sum. If column index is None, each of columns will be sum.

q_n : int, positive integer less than or equal to 16, default: 6
    q_n is the number of significant decimal digit, If the data type is a float, 
    the maximum significant digit is 16. The sum of integer and significant decimal digits should 
    be less than or equal to 16.

"""


class FeldmanVerifiableSumParam(BaseParam):
    def __init__(self, sum_cols=None, q_n=6):
        self.sum_cols = sum_cols
        if sum_cols is None:
            self.sum_cols = []

        self.q_n = q_n

    def check(self):
        if isinstance(self.sum_cols, list):
            for idx in self.sum_cols:
                if not isinstance(idx, int):
                    raise ValueError(f"type mismatch, column_indexes with element {idx}(type is {type(idx)})")

        if not isinstance(self.q_n, int):
            raise ValueError(f"Init param's q_n {self.q_n} not supported, should be int type", type is {type(self.q_n)})

        if self.q_n < 0:
            raise ValueError(f"param's q_n {self.q_n} not supported, should be non-negative int value")
        elif self.q_n > 16:
            raise ValueError(f"param's q_n {self.q_n} not supported, should be less than or equal to 16")
