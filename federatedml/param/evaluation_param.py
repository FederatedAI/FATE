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
from federatedml.util import consts
from federatedml.param.base_param import BaseParam

LOGGER = log_utils.getLogger()


class EvaluateParam(BaseParam):
    """
    Define the evaluation method of binary/multiple classification and regression

    Parameters
    ----------
    eval_type: string, support 'binary' for HomoLR, HeteroLR and Secureboosting. support 'regression' for Secureboosting. 'multi' is not support these version

    pos_label: specify positive label type, can be int, float and str, this depend on the data's label, this parameter effective only for 'binary'

    need_run: bool, default True
        Indicate if this module needed to be run
    """

    def __init__(self, eval_type="binary", pos_label=1, need_run=True):
        super().__init__()
        self.eval_type = eval_type
        self.pos_label = pos_label
        self.need_run = need_run

    def check(self):
        descr = "evaluate param's "
        self.eval_type = self.check_and_change_lower(self.eval_type,
                                                       [consts.BINARY, consts.MULTY, consts.REGRESSION],
                                                       descr)

        if type(self.pos_label).__name__ not in ["str", "float", "int"]:
            raise ValueError(
                "evaluate param's pos_label {} not supported, should be str or float or int type".format(
                    self.pos_label))

        if type(self.need_run).__name__ != "bool":
            raise ValueError(
                "evaluate param's need_run {} not supported, should be bool".format(
                    self.need_run))

        LOGGER.info("Finish evaluation parameter check!")
        return True
