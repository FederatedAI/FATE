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
from federatedml.util import consts
from federatedml.util import LOGGER


class FeatureTransformParam(BaseParam):
    """
    Define method used for expanding column

    Parameters
    ----------

    rules : None or List[object], default: None
        feature transform operations. If None is given, module outputs the original input value without any operation.

    need_run: bool, default: True
        Indicate if this module needed to be run.

    """

    def __init__(self, rules=None, need_run=True):
        super(FeatureTransformParam, self).__init__()
        self.rules = rules
        self.need_run = need_run

    def check(self):
        descr = "feature_transform param's "

        if not isinstance(self.rules, list):
            raise ValueError(f"{descr} rules must be None or list of object. "
                             f"Received {type(self.rules)} instead.")

        BaseParam.check_boolean(self.need_run, descr=descr)

        LOGGER.debug("Finish column expand parameter check!")
        return True
