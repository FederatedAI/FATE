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
    mode: {"standard", "two_step"}
        Switch positive unlabeled learning mode.

    labeling_strategy: {"proportion", "quantity", "probability", "interval"}
        Switch converting unlabeled value strategy.

    threshold_percent: float, default: 0.1
        The threshold percent in proportion strategy.

    threshold_amount: float, default: 10
        The threshold amount in quantity strategy.

    threshold_proba: float, default: 0.9
        The threshold proba in probability strategy.
    """

    def __init__(self, mode="standard", labeling_strategy="proportion",
                 threshold_percent=0.1, threshold_amount=10, threshold_proba=0.9):
        super(PositiveUnlabeledParam, self).__init__()
        self.mode = mode
        self.labeling_strategy = labeling_strategy
        self.threshold_percent = threshold_percent
        self.threshold_amount = threshold_amount
        self.threshold_proba = threshold_proba

    def check(self):
        if self.mode not in ["standard", "two_step"]:
            raise ValueError("mode not supported, it should be 'standard' or 'two_step'")

        if self.labeling_strategy not in ["proportion", "quantity", "probability", "interval"]:
            raise ValueError(
                "labeling_strategy not supported, it should be 'proportion', 'quantity', 'probability' or 'interval'")

        if self.labeling_strategy == "proportion" and type(self.threshold_percent).__name__ != "float":
            raise ValueError("threshold_percent should be a float in proportion strategy")

        if self.labeling_strategy == "quantity" and type(self.threshold_amount).__name__ != "int":
            raise ValueError("threshold_amount should be an integer in quantity strategy")

        if self.labeling_strategy == "probability" and type(self.threshold_proba).__name__ != "float":
            raise ValueError("threshold_proba should be a float in probability strategy")

        if self.mode != "two_step" and self.labeling_strategy == "interval":
            raise ValueError("Interval strategy only adapted to two-step mode")

        return True
