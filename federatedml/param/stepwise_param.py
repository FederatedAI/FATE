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


from federatedml.param.base_param import BaseParam
from federatedml.util import consts


class StepwiseParam(BaseParam):
    """
    Define stepwise params

    Parameters
    ----------
    score_name: str, default: 'AIC'
        Specify which model selection criterion to be used

    mode: str, default: 'Hetero'
        Indicate what mode is current task

    role: str, default: 'Guest'
        Indicate what role is current party

    direction: str, default: 'both'
        Indicate which direction to go for stepwise.
        'forward' means forward selection; 'backward' means elimination; 'both' means possible models of both directions are examined at each step.

    max_step: int, default: '10'
        Specify total number of steps to run before forced stop.

    nvmin: int, default: '2'
        Specify the min subset size of final model, cannot be lower than 2. When nvmin > 2, the final model size may be smaller than nvmin due to max_step limit.

    nvmax: int, default: None
        Specify the max subset size of final model, 2 <= nvmin <= nvmax. The final model size may be larger than nvmax due to max_step limit.

    need_stepwise: bool, default False
        Indicate if this module needed to be run

    """

    def __init__(self, score_name="AIC", mode=consts.HETERO, role=consts.GUEST, direction="both",
                 max_step=10, nvmin=2, nvmax=None, need_stepwise=False):
        super(StepwiseParam, self).__init__()
        self.score_name = score_name
        self.mode = mode
        self.role = role
        self.direction = direction
        self.max_step = max_step
        self.nvmin = nvmin
        self.nvmax = nvmax
        self.need_stepwise = need_stepwise

    def check(self):
        model_param_descr = "stepwise param's"
        self.score_name = self.check_and_change_lower(self.score_name, ["aic", "bic"], model_param_descr)
        self.check_valid_value(self.mode, model_param_descr, valid_values=[consts.HOMO, consts.HETERO])
        self.check_valid_value(self.role, model_param_descr, valid_values=[consts.HOST, consts.GUEST, consts.ARBITER])
        self.direction = self.check_and_change_lower(self.direction, ["forward", "backward", "both"], model_param_descr)
        self.check_positive_integer(self.max_step, model_param_descr)
        self.check_positive_integer(self.nvmin, model_param_descr)
        if self.nvmin < 2:
            raise ValueError(model_param_descr + " nvmin must be no less than 2.")
        if self.nvmax is not None:
            self.check_positive_integer(self.nvmax, model_param_descr)
            if self.nvmin > self.nvmax:
                raise ValueError(model_param_descr + " nvmax must be greater than nvmin.")
        self.check_boolean(self.need_stepwise, model_param_descr)
