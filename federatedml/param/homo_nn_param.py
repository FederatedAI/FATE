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
import copy

from federatedml.param.base_param import BaseParam
from federatedml.param.cross_validation_param import CrossValidationParam
from federatedml.param.one_vs_rest_param import OneVsRestParam
from federatedml.param.predict_param import PredictParam
from federatedml.util import consts


class HomoNNParam(BaseParam):
    """
    Parameters used for Logistic Regression both for Homo mode or Hetero mode.

    Parameters
    ----------

    eps : float, default: 1e-5
        The tolerance of convergence

    optimizer : "Adadelta", "Adagrad", "Adam", "Adamax", "Nadam", "RMSprop", "SGD"
        Optimize method

    batch_size : int, default: -1
        Batch size when updating model. -1 means use all data in a batch. i.e. Not to use mini-batch strategy.

    learning_rate : float, default: 0.01
        Learning rate

    max_iter : int, default: 100
        The maximum iteration for training.

    converge_func : str, 'diff', 'weight_diff' or 'abs', default: 'diff'
        Method used to judge converge or not.
            a)	diff： Use difference of loss between two iterations to judge whether converge.
            b)  weight_diff: Use difference between weights of two consecutive iterations
            c)	abs: Use the absolute value of loss to judge whether converge. i.e. if loss < eps, it is converged.

    """

    def __init__(self, secure_aggregate=True,
                 enable_converge_check=True,
                 config_type="keras",
                 nn_define=None,
                 eps=0.05,
                 optimizer='Adadelta',
                 loss='categorical_crossentropy',
                 metrics=None,
                 aggregate_every_n_epoch=1,
                 batch_size=128,
                 max_iter=10,

                 learning_rate=0.01,
                 converge_func="diff",

                 need_run=True,
                 predict_param=PredictParam(),
                 cv_param=CrossValidationParam(),
                 one_vs_rest_param=OneVsRestParam()):
        super(HomoNNParam, self).__init__()

        self.learning_rate = learning_rate
        self.converge_func = converge_func

        self.need_run = need_run
        self.predict_param = copy.deepcopy(predict_param)
        self.cv_param = copy.deepcopy(cv_param)
        self.one_vs_rest_param = copy.deepcopy(one_vs_rest_param)

        self.max_iter = max_iter
        self.secure_aggregate = secure_aggregate
        self.enable_converge_check = enable_converge_check
        self.aggregate_every_n_epoch = aggregate_every_n_epoch
        self.config_type = config_type
        self.nn_define = nn_define or []
        self.batch_size = batch_size
        self.eps = eps
        self.optimizer = optimizer
        self.loss = loss
        if not metrics:
            self.metrics = []
        elif isinstance(metrics, str):
            self.metrics = [metrics]
        else:
            self.metrics = metrics

    def check(self):
        descr = "homo_nn's"

        attr = self.check_type(descr, "batch_size", int)
        if attr != -1 and attr < consts.MIN_BATCH_SIZE:
            raise ValueError(f"{descr} batch_size={attr} not supported, "
                             f"should be larger than {consts.MIN_BATCH_SIZE} or -1 represent for all data")

        self.check_type(descr, "learning_rate", float)

        attr = self.check_type(descr, "max_iter", int)
        if attr <= 0:
            raise ValueError(f"{descr} max_iter must be greater or equal to 1")

        self.check_in_scope(descr, "converge_func", ["diff", "abs"])
        self.check_in_scope(descr, "optimizer", ["Adadelta", "Adagrad", "Adam", "Adamax", "Nadam", "RMSprop", "SGD"])
        return True

    def check_type(self, descr, name, expected_type):
        attr = getattr(self, name)
        if not isinstance(attr, expected_type):
            raise ValueError(f"{descr} {name} {attr} not supported, should be {expected_type.__name__} type")
        return attr

    def check_in_scope(self, descr, name, scope):
        attr = self.check_type(descr, name, str)
        if attr not in scope:
            raise ValueError(f"{descr} {name} not supported, {name} should be one of {list(*scope)}")
