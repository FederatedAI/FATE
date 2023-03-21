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
from federatedml.util import LOGGER


class EncryptedModeCalculatorParam(BaseParam):
    """
    Define the encrypted_mode_calulator parameters.

    Parameters
    ----------
    mode: {'strict', 'fast', 'balance', 'confusion_opt'}
        encrypted mode, default: strict
    re_encrypted_rate: float or int
        numeric number in [0, 1], use when mode equals to 'balance', default: 1
    """

    def __init__(self, mode="strict", re_encrypted_rate=1):
        self.mode = mode
        self.re_encrypted_rate = re_encrypted_rate

    def check(self):
        descr = "encrypted_mode_calculator param"
        self.mode = self.check_and_change_lower(self.mode,
                                                ["strict", "fast", "balance", "confusion_opt", "confusion_opt_balance"],
                                                descr)

        if self.mode != "strict":
            LOGGER.warning("encrypted_mode_calculator will be remove in later version, "
                           "but in current version user can still use it, but it only supports strict mode, "
                           "other mode will be reset to strict for compatibility")
            self.mode = "strict"

        return True
