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
from pipeline.param import consts
from pipeline.param.base_param import BaseParam


class PositiveUnlabeledParam(BaseParam):
    """
    Parameters used for positive unlabeled.
    ----------
    strategy: {"probability", "quantity", "proportion", "distribution"}
        The strategy of converting unlabeled value.

    threshold: int or float, default: 0.9
        The threshold in labeling strategy.
    """

    def __init__(self, strategy="probability", threshold=0.9):
        super(PositiveUnlabeledParam, self).__init__()
        self.strategy = strategy
        self.threshold = threshold

    def check(self):
        base_descr = "Positive Unlabeled Param's "
        float_descr = "Probability or Proportion Strategy Param's "
        int_descr = "Quantity Strategy Param's "
        numeric_descr = "Distribution Strategy Param's "

        self.check_valid_value(self.strategy, base_descr,
                               [consts.PROBABILITY, consts.QUANTITY, consts.PROPORTION, consts.DISTRIBUTION])

        self.check_defined_type(self.threshold, base_descr, [consts.INT, consts.FLOAT])

        if self.strategy == consts.PROBABILITY or self.strategy == consts.PROPORTION:
            self.check_decimal_float(self.threshold, float_descr)

        if self.strategy == consts.QUANTITY:
            self.check_positive_integer(self.threshold, int_descr)

        if self.strategy == consts.DISTRIBUTION:
            self.check_positive_number(self.threshold, numeric_descr)

        return True
