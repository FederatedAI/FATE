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
from federatedml.param.encrypt_param import EncryptParam
from federatedml.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from federatedml.param.init_model_param import InitParam
from federatedml.param.one_vs_rest_param import OneVsRestParam
from federatedml.param.predict_param import PredictParam
from federatedml.util import consts


class FMInitParam(InitParam):
    def __init__(self, embed_size=10):
        super(FMInitParam, self).__init__()
        self.embed_size = embed_size


class FactorizationParam(BaseParam):
    """
    Parameters used for Factorization Machine.

    Parameters
    ----------
    penalty : str, 'L1' or 'L2'. default: 'L2'

        Penalty method used in FM. Please note that, when using encrypted version in HomoFM,
        'L1' is not supported.

    tol : float, default: 1e-5
        The tolerance of convergence

    alpha : float, default: 1.0
        Regularization strength coefficient.

    optimizer : str, 'sgd', 'rmsprop', 'adam', 'nesterov_momentum_sgd' or 'adagrad', default: 'sgd'
        Optimize method

    batch_size : int, default: -1
        Batch size when updating model. -1 means use all data in a batch. i.e. Not to use mini-batch strategy.

    learning_rate : float, default: 0.01
        Learning rate

    init_param: InitParam object, default: default InitParam object
        Init param method object.

    max_iter : int, default: 100
        The maximum iteration for training.

    early_stop : str, 'diff', 'weight_diff' or 'abs', default: 'diff'
        Method used to judge converge or not.
            a)	diff： Use difference of loss between two iterations to judge whether converge.
            b)  weight_diff: Use difference between weights of two consecutive iterations
            c)	abs: Use the absolute value of loss to judge whether converge. i.e. if loss < eps, it is converged.

    decay: int or float, default: 1
        Decay rate for learning rate. learning rate will follow the following decay schedule.
        lr = lr0/(1+decay*t) if decay_sqrt is False. If decay_sqrt is True, lr = lr0 / sqrt(1+decay*t)
        where t is the iter number.

    decay_sqrt: Bool, default: True
        lr = lr0/(1+decay*t) if decay_sqrt is False, otherwise, lr = lr0 / sqrt(1+decay*t)

    encrypt_param: EncryptParam object, default: default EncryptParam object

    predict_param: PredictParam object, default: default PredictParam object

    cv_param: CrossValidationParam object, default: default CrossValidationParam object

    multi_class: str, 'ovr', default: 'ovr'
        If it is a multi_class task, indicate what strategy to use. Currently, support 'ovr' short for one_vs_rest only.

    """

    def __init__(self, penalty='L2',
                 tol=1e-5, alpha=1.0, optimizer='sgd',
                 batch_size=-1, learning_rate=0.01, init_param=FMInitParam(),
                 max_iter=100, early_stop='diff', encrypt_param=EncryptParam(),
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 decay=1, decay_sqrt=True,
                 multi_class='ovr', validation_freqs=None, clip_gradient=None
                 ):
        super(FactorizationParam, self).__init__()
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
        self.predict_param = copy.deepcopy(predict_param)
        self.cv_param = copy.deepcopy(cv_param)
        self.decay = decay
        self.decay_sqrt = decay_sqrt
        self.multi_class = multi_class
        self.validation_freqs = validation_freqs
        self.clip_gradient = clip_gradient

    def check(self):
        descr = "factorization_param's"

        if self.penalty is None:
            pass
        elif type(self.penalty).__name__ != "str":
            raise ValueError(
                "factorization_param's penalty {} not supported, should be str type".format(self.penalty))
        else:
            self.penalty = self.penalty.upper()
            if self.penalty not in [consts.L1_PENALTY, consts.L2_PENALTY, 'NONE']:
                raise ValueError(
                    "factorization_param's penalty not supported, penalty should be 'L1', 'L2' or 'none'")

        if not isinstance(self.tol, (int, float)):
            raise ValueError(
                "factorization_param's tol {} not supported, should be float type".format(self.tol))

        if self.clip_gradient is None:
            pass
        elif not isinstance(self.clip_gradient, (int, float)):
            raise ValueError(

                "factorization_param's clip_gradient {} not supported, should be float or int type".format(self.clip_gradient))


        if type(self.alpha).__name__ not in ["float", 'int']:
            raise ValueError(
                "factorization_param's alpha {} not supported, should be float or int type".format(self.alpha))

        if type(self.optimizer).__name__ != "str":
            raise ValueError(
                "factorization_param's optimizer {} not supported, should be str type".format(self.optimizer))
        else:
            self.optimizer = self.optimizer.lower()
            if self.optimizer not in ['sgd', 'rmsprop', 'adam', 'adagrad', 'nesterov_momentum_sgd']:
                raise ValueError(
                    "factorization_param's optimizer not supported, optimizer should be"
                    " 'sgd', 'rmsprop', 'adam', 'nesterov_momentum_sgd' or 'adagrad'")

        if self.batch_size != -1:
            if type(self.batch_size).__name__ not in ["int"] \
                    or self.batch_size < consts.MIN_BATCH_SIZE:
                raise ValueError(descr + " {} not supported, should be larger than 10 or "
                                         "-1 represent for all data".format(self.batch_size))


        if not isinstance(self.learning_rate, (float, int)) :
            raise ValueError(
                "factorization_param's learning_rate {} not supported, should be float or int type".format(
                    self.learning_rate))

        if self.learning_rate <= 0:
            raise ValueError(
                "factorization_param's learning_rate {} not supported, should be a positive value".format(
                    self.learning_rate))

        self.init_param.check()

        if type(self.max_iter).__name__ != "int":
            raise ValueError(
                "factorization_param's max_iter {} not supported, should be int type".format(self.max_iter))
        elif self.max_iter <= 0:
            raise ValueError(
                "factorization_param's max_iter must be greater or equal to 1")

        if type(self.early_stop).__name__ != "str":
            raise ValueError(
                "factorization_param's early_stop {} not supported, should be str type".format(
                    self.early_stop))
        else:
            self.early_stop = self.early_stop.lower()
            if self.early_stop not in ['diff', 'abs', 'weight_diff']:
                raise ValueError(
                    "factorization_param's early_stop not supported, converge_func should be"
                    " 'diff' or 'abs'")

        self.encrypt_param.check()
        if self.encrypt_param.method not in [consts.PAILLIER, None]:
            raise ValueError(
                "factorization_param's encrypted method support 'Paillier' or None only")

        if type(self.decay).__name__ not in ["int", 'float']:
            raise ValueError(
                "factorization_param's decay {} not supported, should be 'int' or 'float'".format(
                    self.decay))

        if type(self.decay_sqrt).__name__ not in ['bool']:
            raise ValueError(
                "factorization_param's decay_sqrt {} not supported, should be 'bool'".format(
                    self.decay_sqrt))

        if self.decay < 0:
            raise ValueError(
                "factorization_param's decay must be greater or equal to 0")
        return True


class HomoFactorizationParam(FactorizationParam):
    """
    Parameters
    ----------

    aggregate_iters : int, default: 1
        Indicate how many iterations are aggregated once.

    """
    def __init__(self, penalty='L2',
                 tol=1e-5, alpha=1.0, optimizer='sgd',
                 batch_size=-1, learning_rate=0.01, init_param=FMInitParam(),
                 max_iter=100, early_stop='diff',
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 decay=1, decay_sqrt=True,
                 aggregate_iters=1, multi_class='ovr', validation_freqs=None,
                 clip_gradient=None
                 ):
        super(HomoFactorizationParam, self).__init__(penalty=penalty, tol=tol, alpha=alpha, optimizer=optimizer,
                                                batch_size=batch_size,
                                                learning_rate=learning_rate,
                                                init_param=init_param, max_iter=max_iter, early_stop=early_stop,
                                                predict_param=predict_param,
                                                cv_param=cv_param, multi_class=multi_class,
                                                validation_freqs=validation_freqs,
                                                decay=decay, decay_sqrt=decay_sqrt,
                                                clip_gradient=clip_gradient)
        self.aggregate_iters = aggregate_iters

    def check(self):
        super().check()

        if not isinstance(self.aggregate_iters, int):
            raise ValueError(
                "factorization_param's aggregate_iters {} not supported, should be int type".format(
                    self.aggregate_iters))

        return True


class HeteroFactorizationParam(FactorizationParam):
    def __init__(self, penalty='L2',
                 tol=1e-5, alpha=1.0, optimizer='sgd',
                 batch_size=-1, learning_rate=0.01, init_param=FMInitParam(),
                 max_iter=100, early_stop='diff', encrypt_param=EncryptParam(),
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(),
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 decay=1, decay_sqrt=True,
                 multi_class='ovr', validation_freqs=None,clip_gradient=None
                 ):
        super(HeteroFactorizationParam, self).__init__(penalty=penalty, tol=tol, alpha=alpha, optimizer=optimizer,
                                                  batch_size=batch_size,
                                                  learning_rate=learning_rate, encrypt_param=encrypt_param,
                                                  init_param=init_param, max_iter=max_iter, early_stop=early_stop,
                                                  predict_param=predict_param, cv_param=cv_param,
                                                  decay=decay,
                                                  decay_sqrt=decay_sqrt, multi_class=multi_class,
                                                  validation_freqs=validation_freqs,
                                                  clip_gradient=clip_gradient)
        self.encrypted_mode_calculator_param = encrypted_mode_calculator_param

    def check(self):
        super().check()
        self.encrypted_mode_calculator_param.check()
        if self.encrypt_param.method != consts.PAILLIER:
            raise ValueError("Hetero FM support Paillier encryption mode only")
        return True
