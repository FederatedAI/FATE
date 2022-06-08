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


from pipeline.param.base_param import BaseParam
# from pipeline.param.evaluation_param import EvaluateParam
from pipeline.param import consts


class CrossValidationParam(BaseParam):
    """
    Define cross validation params

    Parameters
    ----------
    n_splits: int, default: 5
        Specify how many splits used in KFold

    mode: str, default: 'Hetero'
        Indicate what mode is current task

    role: {'Guest', 'Host', 'Arbiter'}, default: 'Guest'
        Indicate what role is current party

    shuffle: bool, default: True
        Define whether do shuffle before KFold or not.

    random_seed: int, default: 1
        Specify the random seed for numpy shuffle

    need_cv: bool, default False
        Indicate if this module needed to be run

    output_fold_history: bool, default True
        Indicate whether to output table of ids used by each fold, else return original input data
        returned ids are formatted as: {original_id}#fold{fold_num}#{train/validate}

    history_value_type: {'score', 'instance'}, default score
        Indicate whether to include original instance or predict score in the output fold history,
        only effective when output_fold_history set to True

    """

    def __init__(self, n_splits=5, mode=consts.HETERO, role=consts.GUEST, shuffle=True, random_seed=1,
                 need_cv=False, output_fold_history=True, history_value_type="score"):
        super(CrossValidationParam, self).__init__()
        self.n_splits = n_splits
        self.mode = mode
        self.role = role
        self.shuffle = shuffle
        self.random_seed = random_seed
        # self.evaluate_param = copy.deepcopy(evaluate_param)
        self.need_cv = need_cv
        self.output_fold_history = output_fold_history
        self.history_value_type = history_value_type

    def check(self):
        model_param_descr = "cross validation param's "
        self.check_positive_integer(self.n_splits, model_param_descr)
        self.check_valid_value(self.mode, model_param_descr, valid_values=[consts.HOMO, consts.HETERO])
        self.check_valid_value(self.role, model_param_descr, valid_values=[consts.HOST, consts.GUEST, consts.ARBITER])
        self.check_boolean(self.shuffle, model_param_descr)
        self.check_boolean(self.output_fold_history, model_param_descr)
        self.history_value_type = self.check_and_change_lower(
            self.history_value_type, ["instance", "score"], model_param_descr)
        if self.random_seed is not None:
            self.check_positive_integer(self.random_seed, model_param_descr)
