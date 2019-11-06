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

class LocalTrainParam(BaseParam):
    """
    Define the local train method

    Parameters
    ----------
    model_name: str, sklearn model used to train on local model

    need_cv: bool, default False
        Indicate whether cv is needed

    n_splits: int, default 5
        Number of CV folds if need_cv

    model_opts: dict or None, default None
        Param to be used as input into sklearn model

    shuffle: bool, default False
        Indicate whether to shuffle data for CV

    random_seed: int or None, default 42
        Specify random seed if shuffle

    need_run: bool, default True
        Indicate if this module needed to be run
    """

    def __init__(self, model_name="LogisticRegression", need_cv=False, n_splits=5, model_opts=None,
                 shuffle=False, random_seed=42, need_run=True):
        super(LocalTrainParam, self).__init__()
        self.model_name = model_name
        self.need_cv = need_cv
        self.n_splits = n_splits
        self.model_opts = model_opts
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.need_run = need_run

    def check(self):
        descr = "local train param"

        self.mode = self.check_and_change_lower(self.model_name,
                                                   ["logisticregression"],
                                                   descr)
        self.check_boolean(self.need_cv, descr)
        self.check_boolean(self.need_run, descr)
        self.check_positive_integer(self.n_splits, descr)
        self.check_boolean(self.shuffle, descr)
        if self.model_opts is not None:
            if not isinstance(self.model_opts, dict):
                raise ValueError(descr + " model_opts must be None or dict.")
        if self.random_seed is not None:
            if not isinstance(self.random_seed, int):
                raise ValueError(descr + " randome_state must be None or int.")

        return True
