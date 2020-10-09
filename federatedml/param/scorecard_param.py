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

from arch.api.utils import log_utils

from federatedml.param.base_param import BaseParam
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class ScorecardParam(BaseParam):
    """
    Define method used for transforming prediction score to credit score

    Parameters
    ----------

    method : str, default: 'credit'
        score method, currently only supports "credit"

    offset : int, default: 500
        base score

    factor : int, default: 20
        interval for odds to double

    upper_limit_ratio : int, default: 3
        upper bound for odds ratio, credit score upper bound is upper_limit_ratio * offset

    lower_limit_value : int, default : 0
        lower bound for creidt score

    need_run : bool, default: True
        Indicate if this module needed to be run.

    """

    def __init__(self, method="credit", offset=500, factor=20, upper_limit_ratio=3, lower_limit_value=0, need_run=True):
        super(ScorecardParam, self).__init__()
        self.method = method
        self.offset = offset
        self.factor = factor
        self.upper_limit_ratio = upper_limit_ratio
        self.lower_limit_value = lower_limit_value
        self.need_run = need_run

    def check(self):
        descr = "credit score transform param's "
        if not isinstance(self.method, str):
            raise ValueError(f"{descr}method {self.method} not supported, should be str type")
        else:
            user_input = self.method.lower()
            if user_input == "credit":
                self.method = consts.CREDIT
            else:
                raise ValueError(f"{descr} method {user_input} not supported")

        BaseParam.check_positive_integer(self.offset, descr=descr+"offset ")

        BaseParam.check_positive_integer(self.factor, descr=descr+"factor ")

        BaseParam.check_positive_integer(self.upper_limit_ratio, descr=descr+"upper limit ratio ")

        if type(self.lower_limit_value).__name__ not in ["int", "long"]:
            raise ValueError(f"{descr} lower_limit_value must be int type, received {type(self.lower_limit_value)} instead.")

        BaseParam.check_boolean(self.need_run, descr=descr+"need_run ")

        LOGGER.debug("Finish column expand parameter check!")
        return True
