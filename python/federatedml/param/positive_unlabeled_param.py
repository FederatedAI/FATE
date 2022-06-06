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


class PositiveUnlabeledParam(BaseParam):
    """
    Parameters used for positive unlabeled.
    ----------
    reverse_order: bool, default: True
        Need to reverse predict score or not.

    threshold_percent: float, default: 0.1
        The threshold percent of converting unlabeled value.

    pu_mode: {'standard', 'two_step'}
        Switch positive unlabeled learning mode.
    """

    def __init__(self, reverse_order=True, threshold_percent=0.1, pu_mode="standard"):
        super(PositiveUnlabeledParam, self).__init__()
        self.reverse_order = reverse_order
        self.threshold_percent = threshold_percent
        self.pu_mode = pu_mode

    def check(self):
        if self.threshold_percent is not None and type(self.threshold_percent).__name__ != "float":
            raise ValueError("threshold_percent should be a float")

        BaseParam.check_boolean(self.reverse_order, descr="positive unlabeled param reverse_order")

        if self.pu_mode not in ['standard', 'two_step']:
            raise ValueError("pu_mode not supported, pu_mode should be 'standard' or 'two_step'")

        return True
