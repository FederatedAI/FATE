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

from federatedml.param.base_param import BaseParam, deprecated_param
from federatedml.param.base_param import BaseParam
from federatedml.param.callback_param import CallbackParam
from federatedml.param.encrypt_param import EncryptParam
from federatedml.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from federatedml.param.cross_validation_param import CrossValidationParam
from federatedml.param.init_model_param import InitParam
from federatedml.param.predict_param import PredictParam
from federatedml.param.stepwise_param import StepwiseParam
from federatedml.util import consts


@deprecated_param("validation_freqs", "metrics", "early_stopping_rounds", "use_first_metric_only")
class PoissonParam(BaseParam):
    """
    Parameters used for Poisson Regression.

    Parameters
    ----------
    penalty : {'L2', 'L1'}, default: 'L2'
        Penalty method used in Poisson. Please note that, when using encrypted version in HeteroPoisson,
        'L1' is not supported.

    tol : float, default: 1e-4
        The tolerance of convergence

    alpha : float, default: 1.0
        Regularization strength coefficient.

    optimizer : {'rmsprop', 'sgd', 'adam', 'adagrad'}, default: 'rmsprop'
        Optimize method

    batch_size : int, default: -1
        Batch size when updating model. -1 means use all data in a batch. i.e. Not to use mini-batch strategy.

    learning_rate : float, default: 0.01
        Learning rate

    max_iter : int, default: 20
        The maximum iteration for training.

    init_param: InitParam object, default: default InitParam object
        Init param method object.

    early_stop : str, 'weight_diff', 'diff' or 'abs', default: 'diff'
        Method used to judge convergence.
            a)	diff： Use difference of loss between two iterations to judge whether converge.
            b)  weight_diff: Use difference between weights of two consecutive iterations
            c)	abs: Use the absolute value of loss to judge whether converge. i.e. if loss < eps, it is converged.

    exposure_colname: str or None, default: None
        Name of optional exposure variable in dTable.

    predict_param: PredictParam object, default: default PredictParam object
        predict param

    encrypt_param: EncryptParam object, default: default EncryptParam object
        encrypt param

    encrypted_mode_calculator_param: EncryptedModeCalculatorParam object, default: default EncryptedModeCalculatorParam object
        encrypted mode calculator param

    cv_param: CrossValidationParam object, default: default CrossValidationParam object
        cv param

    stepwise_param: StepwiseParam object, default: default StepwiseParam object
        stepwise param

    decay: int or float, default: 1
        Decay rate for learning rate. learning rate will follow the following decay schedule.
        lr = lr0/(1+decay*t) if decay_sqrt is False. If decay_sqrt is True, lr = lr0 / sqrt(1+decay*t)
        where t is the iter number.

    decay_sqrt: bool, default: True
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
                 tol=1e-4, alpha=1.0, optimizer='rmsprop',
                 batch_size=-1, learning_rate=0.01, init_param=InitParam(),
                 max_iter=20, early_stop='diff',
                 exposure_colname = None, predict_param=PredictParam(),
                 encrypt_param=EncryptParam(),
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(),
                 cv_param=CrossValidationParam(), stepwise_param=StepwiseParam(),
                 decay=1, decay_sqrt=True,
                 validation_freqs=None, early_stopping_rounds=None, metrics=None, use_first_metric_only=False,
                 floating_point_precision=23, callback_param=CallbackParam()):
        super(PoissonParam, self).__init__()
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
        self.exposure_colname = exposure_colname
        self.validation_freqs = validation_freqs
        self.stepwise_param = stepwise_param
        self.early_stopping_rounds = early_stopping_rounds
        self.metrics = metrics or []
        self.use_first_metric_only = use_first_metric_only
        self.floating_point_precision = floating_point_precision
        self.callback_param = copy.deepcopy(callback_param)

    def check(self):
        descr = "poisson_regression_param's "

        if self.penalty is None:
            self.penalty = 'NONE'
        elif type(self.penalty).__name__ != "str":
            raise ValueError(
                descr + "penalty {} not supported, should be str type".format(self.penalty))

        self.penalty = self.penalty.upper()
        if self.penalty not in ['L1', 'L2', 'NONE']:
            raise ValueError(
                "penalty {} not supported, penalty should be 'L1', 'L2' or 'none'".format(self.penalty))

        if type(self.tol).__name__ not in ["int", "float"]:
            raise ValueError(
                descr + "tol {} not supported, should be float type".format(self.tol))

        if type(self.alpha).__name__ not in ["int", "float"]:
            raise ValueError(
                descr + "alpha {} not supported, should be float type".format(self.alpha))

        if type(self.optimizer).__name__ != "str":
            raise ValueError(
                descr + "optimizer {} not supported, should be str type".format(self.optimizer))
        else:
            self.optimizer = self.optimizer.lower()
            if self.optimizer not in ['sgd', 'rmsprop', 'adam', 'adagrad', 'nesterov_momentum_sgd']:
                raise ValueError(
                    descr + "optimizer not supported, optimizer should be"
                    " 'sgd', 'rmsprop', 'adam', 'adagrad' or 'nesterov_momentum_sgd'")

        if type(self.batch_size).__name__ not in ["int", "long"]:
            raise ValueError(
                descr + "batch_size {} not supported, should be int type".format(self.batch_size))
        if self.batch_size != -1:
            if type(self.batch_size).__name__ not in ["int", "long"] \
                or self.batch_size < consts.MIN_BATCH_SIZE:
                raise ValueError(descr + " {} not supported, should be larger than {} or "
                                         "-1 represent for all data".format(self.batch_size, consts.MIN_BATCH_SIZE))

        if type(self.learning_rate).__name__ not in ["int", "float"]:
            raise ValueError(
                descr + "learning_rate {} not supported, should be float type".format(
                    self.learning_rate))

        self.init_param.check()
        if self.encrypt_param.method != consts.PAILLIER:
            raise ValueError(
                descr + "encrypt method supports 'Paillier' only")

        if type(self.max_iter).__name__ != "int":
            raise ValueError(
                descr + "max_iter {} not supported, should be int type".format(self.max_iter))
        elif self.max_iter <= 0:
            raise ValueError(
                descr + "max_iter must be greater or equal to 1")

        if self.exposure_colname is not None:
            if type(self.exposure_colname).__name__ != "str":
                raise ValueError(
                    descr + "exposure_colname {} not supported, should be string type".format(self.exposure_colname))

        if type(self.early_stop).__name__ != "str":
            raise ValueError(
                descr + "early_stop {} not supported, should be str type".format(
                    self.early_stop))
        else:
            self.early_stop = self.early_stop.lower()
            if self.early_stop not in ['diff', 'abs', 'weight_diff']:
                raise ValueError(
                    descr + "early_stop not supported, early_stop should be"
                    " 'diff' or 'abs'")

        self.encrypt_param.check()
        if self.encrypt_param.method != consts.PAILLIER:
            raise ValueError(
                descr + "encrypt method supports 'Paillier' or None only"
            )

        self.encrypted_mode_calculator_param.check()
        if type(self.decay).__name__ not in ["int", "float"]:
            raise ValueError(
                descr + "decay {} not supported, should be 'int' or 'float'".format(self.decay)
            )
        if type(self.decay_sqrt).__name__ not in ["bool"]:
            raise ValueError(
                descr + "decay_sqrt {} not supported, should be 'bool'".format(self.decay)
            )

        self.stepwise_param.check()

        for p in ["early_stopping_rounds", "validation_freqs", "metrics",
                  "use_first_metric_only"]:
            if self._warn_to_deprecate_param(p, "", ""):
                if "callback_param" in self.get_user_feeded():
                    raise ValueError(f"{p} and callback param should not be set simultaneously，"
                                     f"{self._deprecated_params_set}, {self.get_user_feeded()}")
                else:
                    self.callback_param.callbacks = ["PerformanceEvaluate"]
                break

        if self._warn_to_deprecate_param("validation_freqs", descr, "callback_param's 'validation_freqs'"):
            self.callback_param.validation_freqs = self.validation_freqs

        if self._warn_to_deprecate_param("early_stopping_rounds", descr, "callback_param's 'early_stopping_rounds'"):
            self.callback_param.early_stopping_rounds = self.early_stopping_rounds

        if self._warn_to_deprecate_param("metrics", descr, "callback_param's 'metrics'"):
            self.callback_param.metrics = self.metrics

        if self._warn_to_deprecate_param("use_first_metric_only", descr, "callback_param's 'use_first_metric_only'"):
            self.callback_param.use_first_metric_only = self.use_first_metric_only

        if self.floating_point_precision is not None and \
                (not isinstance(self.floating_point_precision, int) or
                 self.floating_point_precision < 0 or self.floating_point_precision > 64):
            raise ValueError("floating point precision should be null or a integer between 0 and 64")
        self.callback_param.check()

        return True
