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


class ScorecardParam(BaseParam):
    """
    Define method used for transforming prediction score to credit score

    Parameters
    ----------

    method : {"credit"}, default: 'credit'
        score method, currently only supports "credit"

    offset : int or float, default: 500
        score baseline

    factor : int or float, default: 20
        scoring step, when odds double, result score increases by this factor

    factor_base : int or float, default: 2
        factor base, value ln(factor_base) is used for calculating result score

    upper_limit_ratio : int or float, default: 3
        upper bound for odds, credit score upper bound is upper_limit_ratio * offset

    lower_limit_value : int or float, default: 0
        lower bound for result score

    need_run : bool, default: True
        Indicate if this module needs to be run.

    """

    def __init__(
            self,
            method="credit",
            offset=500,
            factor=20,
            factor_base=2,
            upper_limit_ratio=3,
            lower_limit_value=0,
            need_run=True):
        super(ScorecardParam, self).__init__()
        self.method = method
        self.offset = offset
        self.factor = factor
        self.factor_base = factor_base
        self.upper_limit_ratio = upper_limit_ratio
        self.lower_limit_value = lower_limit_value
        self.need_run = need_run

    def check(self):
        descr = "scorecard param"
        if not isinstance(self.method, str):
            raise ValueError(f"{descr}method {self.method} not supported, should be str type")
        else:
            user_input = self.method.lower()
            if user_input == "credit":
                self.method = consts.CREDIT
            else:
                raise ValueError(f"{descr} method {user_input} not supported")

        if type(self.offset).__name__ not in ["int", "long", "float"]:
            raise ValueError(f"{descr} offset must be numeric,"
                             f"received {type(self.offset)} instead.")

        if type(self.factor).__name__ not in ["int", "long", "float"]:
            raise ValueError(f"{descr} factor must be numeric,"
                             f"received {type(self.factor)} instead.")

        if type(self.factor_base).__name__ not in ["int", "long", "float"]:
            raise ValueError(f"{descr} factor_base must be numeric,"
                             f"received {type(self.factor_base)} instead.")

        if type(self.upper_limit_ratio).__name__ not in ["int", "long", "float"]:
            raise ValueError(f"{descr} upper_limit_ratio must be numeric,"
                             f"received {type(self.upper_limit_ratio)} instead.")

        if type(self.lower_limit_value).__name__ not in ["int", "long", "float"]:
            raise ValueError(f"{descr} lower_limit_value must be numeric,"
                             f"received {type(self.lower_limit_value)} instead.")

        BaseParam.check_boolean(self.need_run, descr=descr + "need_run ")

        return True
