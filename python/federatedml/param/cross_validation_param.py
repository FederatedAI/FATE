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

import copy

from federatedml.param.base_param import BaseParam
# from federatedml.param.evaluation_param import EvaluateParam
from federatedml.util import consts


class CrossValidationParam(BaseParam):
    """
    Define cross validation params

    Parameters
    ----------
    n_splits: int, default: 5
        Specify how many splits used in KFold

    mode: str, default: 'Hetero'
        Indicate what mode is current task

    role: str, default: 'Guest'
        Indicate what role is current party

    shuffle: bool, default: True
        Define whether do shuffle before KFold or not.

    random_seed: int, default: 1
        Specify the random seed for numpy shuffle

    need_cv: bool, default True
        Indicate if this module needed to be run

    """

    def __init__(self, n_splits=5, mode=consts.HETERO, role=consts.GUEST, shuffle=True, random_seed=1,
                 need_cv=False):
        super(CrossValidationParam, self).__init__()
        self.n_splits = n_splits
        self.mode = mode
        self.role = role
        self.shuffle = shuffle
        self.random_seed = random_seed
        # self.evaluate_param = copy.deepcopy(evaluate_param)
        self.need_cv = need_cv

    def check(self):
        model_param_descr = "cross validation param's "
        self.check_positive_integer(self.n_splits, model_param_descr)
        self.check_valid_value(self.mode, model_param_descr, valid_values=[consts.HOMO, consts.HETERO])
        self.check_valid_value(self.role, model_param_descr, valid_values=[consts.HOST, consts.GUEST, consts.ARBITER])
        self.check_boolean(self.shuffle, model_param_descr)
        if self.random_seed is not None:
            self.check_positive_integer(self.random_seed, model_param_descr)
