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
from pipeline.param import consts


class ColumnExpandParam(BaseParam):
    """
    Define method used for expanding column

    Parameters
    ----------

    append_header : None, str, List[str] default: None
        Name(s) for appended feature(s). If None is given, module outputs the original input value without any operation.

    method : str, default: 'manual'
        If method is 'manual', use user-specified `fill_value` to fill in new features.

    fill_value : int, float, str, List[int], List[float], List[str] default: 1e-8
        Used for filling expanded feature columns. If given a list, length of the list must match that of `append_header`

    need_run: bool, default: True
        Indicate if this module needed to be run.

    """

    def __init__(self, append_header=None, method="manual",
                 fill_value=consts.FLOAT_ZERO, need_run=True):
        super(ColumnExpandParam, self).__init__()
        self.append_header = [] if append_header is None else append_header
        self.method = method
        self.fill_value = fill_value
        self.need_run = need_run

    def check(self):
        descr = "column_expand param's "
        if not isinstance(self.method, str):
            raise ValueError(f"{descr}method {self.method} not supported, should be str type")
        else:
            user_input = self.method.lower()
            if user_input == "manual":
                self.method = consts.MANUAL
            else:
                raise ValueError(f"{descr} method {user_input} not supported")

        BaseParam.check_boolean(self.need_run, descr=descr)

        if not isinstance(self.append_header, list):
            raise ValueError(f"{descr} append_header must be None or list of str. "
                             f"Received {type(self.append_header)} instead.")
        for feature_name in self.append_header:
            BaseParam.check_string(feature_name, descr+"append_header values")

        if isinstance(self.fill_value, list):
            if len(self.append_header) != len(self.fill_value):
                raise ValueError(
                    f"{descr} `fill value` is set to be list, "
                    f"and param `append_header` must also be list of the same length.")
        else:
            self.fill_value = [self.fill_value]
        for value in self.fill_value:
            if type(value).__name__ not in ["float", "int", "long", "str"]:
                raise ValueError(
                    f"{descr} fill value(s) must be float, int, or str. Received type {type(value)} instead.")

        return True
