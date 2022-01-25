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

from pipeline.param.glm_param import LinearModelParam
from pipeline.param.callback_param import CallbackParam
from pipeline.param.encrypt_param import EncryptParam
from pipeline.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from pipeline.param.cross_validation_param import CrossValidationParam
from pipeline.param.init_model_param import InitParam
from pipeline.param.sqn_param import StochasticQuasiNewtonParam
from pipeline.param.stepwise_param import StepwiseParam
from pipeline.param import consts


class LinearParam(LinearModelParam):
    """
    Parameters used for Linear Regression.

    Parameters
    ----------
    penalty : {'L2' or 'L1'}
        Penalty method used in LinR. Please note that, when using encrypted version in HeteroLinR,
        'L1' is not supported.

    tol : float, default: 1e-4
        The tolerance of convergence

    alpha : float, default: 1.0
        Regularization strength coefficient.

    optimizer : {'sgd', 'rmsprop', 'adam', 'sqn', 'adagrad'}
        Optimize method

    batch_size : int, default: -1
        Batch size when updating model. -1 means use all data in a batch. i.e. Not to use mini-batch strategy.

    learning_rate : float, default: 0.01
        Learning rate

    max_iter : int, default: 20
        The maximum iteration for training.

    init_param: InitParam object, default: default InitParam object
        Init param method object.

    early_stop : {'diff', 'abs', 'weight_dff'}
        Method used to judge convergence.
            a)	diff： Use difference of loss between two iterations to judge whether converge.
            b)	abs: Use the absolute value of loss to judge whether converge. i.e. if loss < tol, it is converged.
            c)  weight_diff: Use difference between weights of two consecutive iterations

    encrypt_param: EncryptParam object, default: default EncryptParam object
        encrypt param

    encrypted_mode_calculator_param: EncryptedModeCalculatorParam object, default: default EncryptedModeCalculatorParam object
        encrypted mode calculator param

    cv_param: CrossValidationParam object, default: default CrossValidationParam object
        cv param

    decay: int or float, default: 1
        Decay rate for learning rate. learning rate will follow the following decay schedule.
        lr = lr0/(1+decay*t) if decay_sqrt is False. If decay_sqrt is True, lr = lr0 / sqrt(1+decay*t)
        where t is the iter number.

    decay_sqrt: Bool, default: True
        lr = lr0/(1+decay*t) if decay_sqrt is False, otherwise, lr = lr0 / sqrt(1+decay*t)

    validation_freqs: int, list, tuple, set, or None
        validation frequency during training, required when using early stopping.
        The default value is None, 1 is suggested. You can set it to a number larger than 1 in order to speed up training by skipping validation rounds.
        When it is larger than 1, a number which is divisible by "max_iter" is recommended, otherwise, you will miss the validation scores of the last training iteration.

    early_stopping_rounds: int, default: None
        If positive number specified, at every specified training rounds, program checks for early stopping criteria.
        Validation_freqs must also be set when using early stopping.

    metrics: list or None, default: None
        Specify which metrics to be used when performing evaluation during training process. If metrics have not improved at early_stopping rounds, trianing stops before convergence.
        If set as empty, default metrics will be used. For regression tasks, default metrics are ['root_mean_squared_error', 'mean_absolute_error']

    use_first_metric_only: bool, default: False
        Indicate whether to use the first metric in `metrics` as the only criterion for early stopping judgement.

    floating_point_precision: None or integer
        if not None, use floating_point_precision-bit to speed up calculation,
        e.g.: convert an x to round(x * 2**floating_point_precision) during Paillier operation, divide
                the result by 2**floating_point_precision in the end.
    callback_param: CallbackParam object
        callback param

    """

    def __init__(self, penalty='L2',
                 tol=1e-4, alpha=1.0, optimizer='sgd',
                 batch_size=-1, learning_rate=0.01, init_param=InitParam(),
                 max_iter=20, early_stop='diff',
                 encrypt_param=EncryptParam(), sqn_param=StochasticQuasiNewtonParam(),
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(),
                 cv_param=CrossValidationParam(), decay=1, decay_sqrt=True, validation_freqs=None,
                 early_stopping_rounds=None, stepwise_param=StepwiseParam(), metrics=None, use_first_metric_only=False,
                 floating_point_precision=23, callback_param=CallbackParam()):
        super(LinearParam, self).__init__(penalty=penalty, tol=tol, alpha=alpha, optimizer=optimizer,
                                          batch_size=batch_size, learning_rate=learning_rate,
                                          init_param=init_param, max_iter=max_iter, early_stop=early_stop,
                                          encrypt_param=encrypt_param, cv_param=cv_param, decay=decay,
                                          decay_sqrt=decay_sqrt, validation_freqs=validation_freqs,
                                          early_stopping_rounds=early_stopping_rounds,
                                          stepwise_param=stepwise_param, metrics=metrics,
                                          use_first_metric_only=use_first_metric_only,
                                          floating_point_precision=floating_point_precision,
                                          callback_param=callback_param)
        self.sqn_param = copy.deepcopy(sqn_param)
        self.encrypted_mode_calculator_param = copy.deepcopy(encrypted_mode_calculator_param)

    def check(self):
        descr = "linear_regression_param's "
        super(LinearParam, self).check()
        if self.optimizer not in ['sgd', 'rmsprop', 'adam', 'adagrad', 'sqn']:
            raise ValueError(
                descr + "optimizer not supported, optimizer should be"
                        " 'sgd', 'rmsprop', 'adam', 'sqn' or 'adagrad'")
        self.sqn_param.check()
        if self.encrypt_param.method != consts.PAILLIER:
            raise ValueError(
                descr + "encrypt method supports 'Paillier' only")
        return True
