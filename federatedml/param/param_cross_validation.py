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

from federatedml.util import consts


class CrossValidationParam(object):
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

    """

    def __init__(self, n_splits=5, mode=consts.HETERO, role=consts.GUEST, shuffle=True, random_seed=1,
                 evaluate_param=EvaluateParam()):
        self.n_splits = n_splits
        self.mode = mode
        self.role = role
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.evaluate_param = copy.deepcopy(evaluate_param)


class EvaluateParam(object):
    """
    Define the evaluation method of binary/multiple classification and regression

    Parameters
    ----------
    metrics: A list of evaluate index. Support 'auc', 'ks', 'lift', 'precision' ,'recall' and 'accuracy', 'explain_variance',
            'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error','median_absolute_error','r2_score','root_mean_squared_error'.
            For example, metrics can be set as ['auc', 'precision', 'recall'], then the results of these indexes will be output.

    classi_type: string, support 'binary' for HomoLR, HeteroLR and Secureboosting. support 'regression' for Secureboosting. 'multi' is not support these version

    pos_label: specify positive label type, can be int, float and str, this depend on the data's label, this parameter effective only for 'binary'

    thresholds: A list of threshold. Specify the threshold use to separate positive and negative class. for example [0.1, 0.3,0.5], this parameter effective only for 'binary'
    """

    def __init__(self, metrics=None, classi_type="binary", pos_label=1, thresholds=None):
        if metrics is None:
            metrics = []
        self.metrics = metrics
        self.classi_type = classi_type
        self.pos_label = pos_label
        if thresholds is None:
            thresholds = [0.5]

        self.thresholds = thresholds
