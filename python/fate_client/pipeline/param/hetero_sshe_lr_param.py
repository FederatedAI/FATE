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

from pipeline.param.base_param import BaseParam
from pipeline.param.cross_validation_param import CrossValidationParam
from pipeline.param.callback_param import CallbackParam
from pipeline.param.encrypt_param import EncryptParam
from pipeline.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from pipeline.param.init_model_param import InitParam
from pipeline.param.predict_param import PredictParam
from pipeline.param import consts


class LogisticRegressionParam(BaseParam):
    """
    Parameters used for Hetero SSHE Logistic Regression

    Parameters
    ----------
    penalty : str, 'L1', 'L2' or None. default: None
        Penalty method used in LR. If it is not None, weights are required to be reconstruct every iter.

    tol : float, default: 1e-4
        The tolerance of convergence

    alpha : float, default: 1.0
        Regularization strength coefficient.

    optimizer : str, 'sgd', 'rmsprop', 'adam', 'nesterov_momentum_sgd', or 'adagrad', default: 'sgd'
        Optimize method

    batch_size : int, default: -1
        Batch size when updating model. -1 means use all data in a batch. i.e. Not to use mini-batch strategy.

    learning_rate : float, default: 0.01
        Learning rate

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
        encrypt param

    predict_param: PredictParam object, default: default PredictParam object
        predict param

    cv_param: CrossValidationParam object, default: default CrossValidationParam object
        cv param

    multi_class: str, 'ovr', default: 'ovr'
        If it is a multi_class task, indicate what strategy to use. Currently, support 'ovr' short for one_vs_rest only.

    reveal_strategy: str, "respectively", "encrypted_reveal_in_host", default: "respectively"
        "respectively": Means guest and host can reveal their own part of weights only.
        "encrypted_reveal_in_host": Means host can be revealed his weights in encrypted mode, and guest can be revealed in normal mode.

    reveal_every_iter: bool, default: True
        Whether reconstruct model weights every iteration. If so, Regularization is available.
        The performance will be better as well since the algorithm process is simplified.

    """

    def __init__(self, penalty=None,
                 tol=1e-4, alpha=1.0, optimizer='sgd',
                 batch_size=-1, learning_rate=0.01, init_param=InitParam(),
                 max_iter=100, early_stop='diff', encrypt_param=EncryptParam(),
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 decay=1, decay_sqrt=True,
                 multi_class='ovr', use_mix_rand=True,
                 reveal_strategy="respectively",
                 reveal_every_iter=True,
                 callback_param=CallbackParam(),
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam()
                 ):
        super(LogisticRegressionParam, self).__init__()
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
        self.decay = decay
        self.decay_sqrt = decay_sqrt
        self.multi_class = multi_class
        self.use_mix_rand = use_mix_rand
        self.reveal_strategy = reveal_strategy
        self.reveal_every_iter = reveal_every_iter
        self.callback_param = copy.deepcopy(callback_param)
        self.cv_param = cv_param
        self.encrypted_mode_calculator_param = copy.deepcopy(encrypted_mode_calculator_param)

    def check(self):
        descr = "logistic_param's"

        if self.penalty is None:
            pass
        elif type(self.penalty).__name__ != "str":
            raise ValueError(
                "logistic_param's penalty {} not supported, should be str type".format(self.penalty))
        else:
            self.penalty = self.penalty.upper()
            if self.penalty not in [consts.L1_PENALTY, consts.L2_PENALTY]:
                raise ValueError(
                    "logistic_param's penalty not supported, penalty should be 'L1', 'L2' or 'none'")
            if not self.reveal_every_iter:
                if self.penalty not in [consts.L2_PENALTY]:
                    raise ValueError(
                        f"penalty should be 'L2' or 'none', when reveal_every_iter is False"
                    )

        if not isinstance(self.tol, (int, float)):
            raise ValueError(
                "logistic_param's tol {} not supported, should be float type".format(self.tol))

        if type(self.alpha).__name__ not in ["float", 'int']:
            raise ValueError(
                "logistic_param's alpha {} not supported, should be float or int type".format(self.alpha))

        if type(self.optimizer).__name__ != "str":
            raise ValueError(
                "logistic_param's optimizer {} not supported, should be str type".format(self.optimizer))
        else:
            self.optimizer = self.optimizer.lower()
            if self.reveal_every_iter:
                if self.optimizer not in ['sgd', 'rmsprop', 'adam', 'adagrad', 'nesterov_momentum_sgd']:
                    raise ValueError(
                        "When reveal_every_iter is True, "
                        "sshe logistic_param's optimizer not supported, optimizer should be"
                        " 'sgd', 'rmsprop', 'adam', 'nesterov_momentum_sgd', or 'adagrad'")
            else:
                if self.optimizer not in ['sgd', 'nesterov_momentum_sgd']:
                    raise ValueError("When reveal_every_iter is False, "
                                     "sshe logistic_param's optimizer not supported, optimizer should be"
                                     " 'sgd', 'nesterov_momentum_sgd'")

        if self.batch_size != -1:
            if type(self.batch_size).__name__ not in ["int"] \
                    or self.batch_size < consts.MIN_BATCH_SIZE:
                raise ValueError(descr + " {} not supported, should be larger than {} or "
                                         "-1 represent for all data".format(self.batch_size, consts.MIN_BATCH_SIZE))

        if not isinstance(self.learning_rate, (float, int)):
            raise ValueError(
                "logistic_param's learning_rate {} not supported, should be float or int type".format(
                    self.learning_rate))

        self.init_param.check()

        if type(self.max_iter).__name__ != "int":
            raise ValueError(
                "logistic_param's max_iter {} not supported, should be int type".format(self.max_iter))
        elif self.max_iter <= 0:
            raise ValueError(
                "logistic_param's max_iter must be greater or equal to 1")

        if type(self.early_stop).__name__ != "str":
            raise ValueError(
                "logistic_param's early_stop {} not supported, should be str type".format(
                    self.early_stop))
        else:
            self.early_stop = self.early_stop.lower()
            if self.early_stop not in ['diff', 'abs', 'weight_diff']:
                raise ValueError(
                    "logistic_param's early_stop not supported, converge_func should be"
                    " 'diff', 'weight_diff' or 'abs'")

        self.encrypt_param.check()
        self.predict_param.check()
        if self.encrypt_param.method not in [consts.PAILLIER, None]:
            raise ValueError(
                "logistic_param's encrypted method support 'Paillier' or None only")

        if type(self.decay).__name__ not in ["int", 'float']:
            raise ValueError(
                "logistic_param's decay {} not supported, should be 'int' or 'float'".format(
                    self.decay))

        if type(self.decay_sqrt).__name__ not in ['bool']:
            raise ValueError(
                "logistic_param's decay_sqrt {} not supported, should be 'bool'".format(
                    self.decay_sqrt))

        if self.callback_param.validation_freqs is not None:
            if type(self.callback_param.validation_freqs).__name__ not in ["int", "list", "tuple", "set"]:
                raise ValueError(
                    "validation strategy param's validate_freqs's type not supported ,"
                    " should be int or list or tuple or set"
                )
            if type(self.callback_param.validation_freqs).__name__ == "int" and \
                    self.callback_param.validation_freqs <= 0:
                raise ValueError("validation strategy param's validate_freqs should greater than 0")
            if self.reveal_every_iter is False:
                raise ValueError(f"When reveal_every_iter is False, validation every iter"
                                 f" is not supported.")

        if self.callback_param.early_stopping_rounds is None:
            pass
        elif isinstance(self.callback_param.early_stopping_rounds, int):
            if self.callback_param.early_stopping_rounds < 1:
                raise ValueError("early stopping rounds should be larger than 0 when it's integer")
            if self.callback_param.validation_freqs is None:
                raise ValueError("validation freqs must be set when early stopping is enabled")

        if self.callback_param.metrics is not None and \
                not isinstance(self.callback_param.metrics, list):
            raise ValueError("metrics should be a list")

        if not isinstance(self.callback_param.use_first_metric_only, bool):
            raise ValueError("use_first_metric_only should be a boolean")

        self.reveal_strategy = self.reveal_strategy.lower()
        self.check_valid_value(self.reveal_strategy, descr, ["respectively", "encrypted_reveal_in_host"])

        if self.reveal_strategy == "encrypted_reveal_in_host" and self.reveal_every_iter:
            raise PermissionError("reveal strategy: encrypted_reveal_in_host mode is not allow to reveal every iter.")
        self.check_boolean(self.reveal_every_iter, descr)
        self.callback_param.check()
        self.cv_param.check()
        return True
