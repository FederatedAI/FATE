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
from federatedml.param.encrypt_param import EncryptParam
from federatedml.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from federatedml.param.cross_validation_param import CrossValidationParam
from federatedml.param.init_model_param import InitParam
from federatedml.param.predict_param import PredictParam
from federatedml.util import consts


class LinearParam(BaseParam):
    """
    Parameters used for Linear Regression.

    Parameters
    ----------
    penalty : str, 'L1' or 'L2'. default: 'L2'
        Penalty method used in LR. Please note that, when using encrypted version in HomoLR,
        'L1' is not supported.

    tol : float, default: 1e-5
        The tolerance of convergence

    alpha : float, default: 1.0
        Regularization strength coefficient.

    optimizer : str, 'sgd', 'rmsprop', 'adam' or 'adagrad', default: 'sgd'
        Optimize method

    party_weight : int or float, default: 1
        Required in Homo LR. Setting the weight of model updated for this party.
        The higher weight set, the higher influence made for this party when updating model.

    batch_size : int, default: -1
        Batch size when updating model. -1 means use all data in a batch. i.e. Not to use mini-batch strategy.

    learning_rate : float, default: 0.01
        Learning rate

    max_iter : int, default: 100
        The maximum iteration for training.

    early_stop : str, 'diff' or 'abs' or 'weight_dff', default: 'diff'
        Method used to judge converge or not.
            a)	diff： Use difference of loss between two iterations to judge whether converge.
            b)	abs: Use the absolute value of loss to judge whether converge. i.e. if loss < tol, it is converged.
            c)  weight_diff: Use difference between weights of two consecutive iterations

    decay: int or float, default: 1
        Decay rate for learning rate. learning rate will follow the following decay schedule:
        lr = lr0/(1+decay*t) if decay_sqrt is False. If decay_sqrt is True, lr = lr0 / sqrt(1+decay*t)
        where t is the iter number.

    """

    def __init__(self, penalty='L2',
                 tol=1e-5, alpha=1.0, optimizer='sgd',
                 batch_size=-1, learning_rate=0.01, init_param=InitParam(),
                 max_iter=100, early_stop='diff', predict_param=PredictParam(),
                 encrypt_param=EncryptParam(),
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(),
                 cv_param=CrossValidationParam(), decay=1, decay_sqrt=True, validation_freqs=None):
        super(LinearParam, self).__init__()
        self.penalty = penalty
        self.tol = tol
        self.alpha = alpha
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_param = copy.deepcopy(init_param)
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.encrypt_param = encrypt_param
        self.encrypted_mode_calculator_param = copy.deepcopy(encrypted_mode_calculator_param)
        self.cv_param = copy.deepcopy(cv_param)
        self.predict_param = copy.deepcopy(predict_param)
        self.decay = decay
        self.decay_sqrt = decay_sqrt
        self.validation_freqs = validation_freqs

    def check(self):
        descr = "linear_param's"

        if type(self.penalty).__name__ != "str":
            raise ValueError(
                "linear_param's penalty {} not supported, should be str type".format(self.penalty))
        else:
            self.penalty = self.penalty.upper()
            if self.penalty not in ['L1', 'L2', 'NONE']:
                raise ValueError(
                    "linear_param's penalty not supported, penalty should be 'L1', 'L2' or 'none'")

        if type(self.tol).__name__ not in ["int", "float"]:
            raise ValueError(
                "linear_param's tol {} not supported, should be float type".format(self.tol))

        if type(self.alpha).__name__ not in ["int", "float"]:
            raise ValueError(
                "linear_param's alpha {} not supported, should be float type".format(self.alpha))

        if type(self.optimizer).__name__ != "str":
            raise ValueError(
                "linear_param's optimizer {} not supported, should be str type".format(self.optimizer))
        else:
            self.optimizer = self.optimizer.lower()
            if self.optimizer not in ['sgd', 'rmsprop', 'adam', 'adagrad']:
                raise ValueError(
                    "linear_param's optimizer not supported, optimizer should be"
                    " 'sgd', 'rmsprop', 'adam' or 'adagrad'")

        if type(self.batch_size).__name__ != "int":
            raise ValueError(
                "linear_param's batch_size {} not supported, should be int type".format(self.batch_size))
        if self.batch_size != -1:
            if type(self.batch_size).__name__ not in ["int", "long"] \
                    or self.batch_size < consts.MIN_BATCH_SIZE:
                raise ValueError(descr + " {} not supported, should be larger than {} or "
                                         "-1 represent for all data".format(consts.MIN_BATCH_SIZE, self.batch_size))

        if type(self.learning_rate).__name__ != "float":
            raise ValueError(
                "linear_param's learning_rate {} not supported, should be float type".format(
                    self.learning_rate))

        self.init_param.check()

        if type(self.max_iter).__name__ != "int":
            raise ValueError(
                "linear_param's max_iter {} not supported, should be int type".format(self.max_iter))
        elif self.max_iter <= 0:
            raise ValueError(
                "linear_param's max_iter must be greater or equal to 1")

        if type(self.early_stop).__name__ != "str":
            raise ValueError(
                "linear_param's early_stop {} not supported, should be str type".format(
                    self.early_stop))
        else:
            self.early_stop = self.early_stop.lower()
            if self.early_stop not in ['diff', 'abs', 'weight_diff']:
                raise ValueError(
                    "linear_param's early_stop not supported, early_stop should be"
                    " 'weight_diff', 'diff' or 'abs'")

        self.encrypt_param.check()
        if self.encrypt_param.method != consts.PAILLIER:
            raise ValueError(
                "Linear regression's encrypt method supports 'Paillier' or None only"
            )

        if type(self.decay).__name__ not in ["int", "float"]:
            raise ValueError(
                "regression param's decay {} not supported, should be 'int' or 'float'".format(self.decay)
            )
        if type(self.decay_sqrt).__name__ not in ["bool"]:
            raise ValueError(
                "regression param's decay_sqrt {} not supported, should be 'bool'".format(self.decay)
            )
        if self.validation_freqs is not None:
            if type(self.validation_freqs).__name__ not in ["int", "list", "tuple", "set"]:
                raise ValueError(
                    "validation strategy param's validate_freqs's type not supported , should be int or list or tuple or set"
                )
            if type(self.validation_freqs).__name__ == "int" and self.validation_freqs <= 0:
                raise ValueError("validation strategy param's validate_freqs should greater than 0")

        return True
