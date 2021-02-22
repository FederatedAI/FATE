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
import numpy as np


class FixedPointEncoder(object):
    def __init__(self, fixpoint_precision=2**23):
        self._fixpoint_precision = fixpoint_precision

    def encode(self, obj):
        if isinstance(obj, np.ndarray):
            fixed_obj = np.round(obj * self._fixpoint_precision, 0).astype(int)
        elif isinstance(obj, list):
            fixed_obj = np.round(np.array(obj) * self._fixpoint_precision, 0).astype(int).to_list()
        else:
            raise ValueError("FixPointEncoder Not support type {}".format(type(obj)))

        return fixed_obj

    def decode(self, obj):
        if isinstance(obj, np.ndarray):
            decode_obj = obj / self._fixpoint_precision
        elif isinstance(obj, list):
            decode_obj = (np.array(obj) / self._fixpoint_precision).to_list()
        else:
            raise ValueError("FixPointEncoder Not support type {}".format(type(obj)))

        return decode_obj
