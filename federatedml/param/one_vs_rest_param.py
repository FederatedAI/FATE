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

LOGGER = log_utils.getLogger()


class OneVsRestParam(BaseParam):
    """
    Define the one_vs_rest parameters.

    Parameters
    ----------
    has_arbiter: bool. For some algorithm, may not has arbiter, for instances, secureboost of FATE,
                     for these algorithms, it should be set to false.
                default true
    """

    def __init__(self, need_one_vs_rest=False, has_arbiter=True):
        super().__init__()
        self.need_one_vs_rest = need_one_vs_rest
        self.has_arbiter = has_arbiter

    def check(self):
        if type(self.has_arbiter).__name__ != "bool":
            raise ValueError(
                "one_vs_rest param's has_arbiter {} not supported, should be bool type".format(
                    self.has_arbiter))

        LOGGER.debug("Finish one_vs_rest parameter check!")
        return True
