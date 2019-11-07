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

class LocalBaselineParam(BaseParam):
    """
    Define the local baseline model param

    Parameters
    ----------
    model_name: str, sklearn model used to train on baseline model

    model_opts: dict or none, default None
        Param to be used as input into baseline model

    need_run: bool, default True
        Indicate if this module needed to be run
    """

    def __init__(self, model_name="LogisticRegression", model_opts=None, need_run=True):
        super(LocalBaselineParam, self).__init__()
        self.model_name = model_name
        self.model_opts = model_opts
        self.need_run = need_run

    def check(self):
        descr = "local baseline param"

        self.mode = self.check_and_change_lower(self.model_name,
                                                   ["logisticregression"],
                                                   descr)
        self.check_boolean(self.need_run, descr)
        if self.model_opts is not None:
            if not isinstance(self.model_opts, dict):
                raise ValueError(descr + " model_opts must be None or dict.")
        if self.model_opts is None:
            self.model_opts = {}

        return True
