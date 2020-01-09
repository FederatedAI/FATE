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

import numpy as np

from arch.api.utils import log_utils
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class _Optimizer(object):
    def __init__(self, learning_rate, alpha, penalty, decay, decay_sqrt):
        self.learning_rate = learning_rate
        self.iters = 0
        self.alpha = alpha
        self.penalty = penalty
        self.decay = decay
        self.decay_sqrt = decay_sqrt

    def decay_learning_rate(self):
        if self.decay_sqrt:
            lr = self.learning_rate / np.sqrt(1 + self.decay * self.iters)
        else:
            lr = self.learning_rate / (1 + self.decay * self.iters)
        return lr

    @property
    def shrinkage_val(self):
        this_step_size = self.learning_rate / np.sqrt(self.iters)
        return self.alpha * this_step_size

    def set_iters(self, iters):
        self.iters = iters

    def apply_gradients(self, grad):
        raise NotImplementedError("Should not call here")

    def _l1_updator(self, model_weights: LinearModelWeights, gradient):
        coef_ = model_weights.coef_
        if model_weights.fit_intercept:
            gradient_without_intercept = gradient[: -1]
        else:
            gradient_without_intercept = gradient

        new_weights = np.sign(coef_ - gradient_without_intercept) * np.maximum(0, np.abs(
            coef_ - gradient_without_intercept) - self.shrinkage_val)

        if model_weights.fit_intercept:
            new_weights = np.append(new_weights, model_weights.intercept_)
            new_weights[-1] -= gradient[-1]
        new_param = LinearModelWeights(new_weights, model_weights.fit_intercept)
        LOGGER.debug("In _l1_updator, original weight: {}, new_weights: {}".format(
            model_weights.unboxed, new_weights
        ))
        return new_param

    def _l2_updator(self, lr_weights: LinearModelWeights, gradient):
        """
        For l2 regularization, the regular term has been added in gradients.
        """

        new_weights = lr_weights.unboxed - gradient
        new_param = LinearModelWeights(new_weights, lr_weights.fit_intercept)

        return new_param

    def add_regular_to_grad(self, grad, lr_weights):

        if self.penalty == consts.L2_PENALTY:
            if lr_weights.fit_intercept:
                gradient_without_intercept = grad[: -1]
                gradient_without_intercept += self.alpha * lr_weights.coef_
                new_grad = np.append(gradient_without_intercept, grad[-1])
            else:
                new_grad = grad + self.alpha * lr_weights.coef_
        else:
            new_grad = grad

        return new_grad

    def regularization_update(self, model_weights: LinearModelWeights, grad):
        if self.penalty == consts.L1_PENALTY:
            model_weights = self._l1_updator(model_weights, grad)
        elif self.penalty == consts.L2_PENALTY:
            model_weights = self._l2_updator(model_weights, grad)
        else:
            new_vars = model_weights.unboxed - grad
            model_weights = LinearModelWeights(new_vars, model_weights.fit_intercept)
        return model_weights

    def __l1_loss_norm(self, model_weights: LinearModelWeights):
        coef_ = model_weights.coef_
        loss_norm = np.sum(self.alpha * np.abs(coef_))
        return loss_norm

    def __l2_loss_norm(self, model_weights: LinearModelWeights):
        coef_ = model_weights.coef_
        loss_norm = 0.5 * self.alpha * np.dot(coef_, coef_)
        return loss_norm

    def loss_norm(self, model_weights: LinearModelWeights):
        if self.penalty == consts.L1_PENALTY:
            loss_norm_value = self.__l1_loss_norm(model_weights)
        elif self.penalty == consts.L2_PENALTY:
            loss_norm_value = self.__l2_loss_norm(model_weights)
        else:
            loss_norm_value = None
        return loss_norm_value

    def hess_vector_norm(self, delta_s: LinearModelWeights):
        if self.penalty == consts.L1_PENALTY:
            return LinearModelWeights(np.zeros_like(delta_s.unboxed), fit_intercept=delta_s.fit_intercept)
        elif self.penalty == consts.L2_PENALTY:
            return LinearModelWeights(self.alpha * np.array(delta_s.unboxed), fit_intercept=delta_s.fit_intercept)
        else:
            return LinearModelWeights(np.zeros_like(delta_s.unboxed), fit_intercept=delta_s.fit_intercept)

    def update_model(self, model_weights: LinearModelWeights, grad, has_applied=True):

        if not has_applied:
            grad = self.add_regular_to_grad(grad, model_weights)
            delta_grad = self.apply_gradients(grad)
        else:
            delta_grad = grad
        model_weights = self.regularization_update(model_weights, delta_grad)
        return model_weights


class _SgdOptimizer(_Optimizer):
    def apply_gradients(self, grad):
        learning_rate = self.decay_learning_rate()

        delta_grad = learning_rate * grad
        LOGGER.debug("In sgd optimizer, learning_rate: {}, delta_grad: {}".format(learning_rate, delta_grad))

        return delta_grad


class _RMSPropOptimizer(_Optimizer):
    def __init__(self, learning_rate, alpha, penalty, decay, decay_sqrt):
        super().__init__(learning_rate, alpha, penalty, decay, decay_sqrt)
        self.rho = 0.99
        self.opt_m = None

    def apply_gradients(self, grad):
        learning_rate = self.decay_learning_rate()

        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)

        self.opt_m = self.rho * self.opt_m + (1 - self.rho) * np.square(grad)
        self.opt_m = np.array(self.opt_m, dtype=np.float64)
        delta_grad = learning_rate * grad / np.sqrt(self.opt_m + 1e-6)
        return delta_grad


class _AdaGradOptimizer(_Optimizer):
    def __init__(self, learning_rate, alpha, penalty, decay, decay_sqrt):
        super().__init__(learning_rate, alpha, penalty, decay, decay_sqrt)
        self.opt_m = None

    def apply_gradients(self, grad):
        learning_rate = self.decay_learning_rate()

        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)
        self.opt_m = self.opt_m + np.square(grad)
        self.opt_m = np.array(self.opt_m, dtype=np.float64)
        delta_grad = learning_rate * grad / (np.sqrt(self.opt_m) + 1e-7)
        return delta_grad


class _NesterovMomentumSGDOpimizer(_Optimizer):
    def __init__(self, learning_rate, alpha, penalty, decay, decay_sqrt):
        super().__init__(learning_rate, alpha, penalty, decay, decay_sqrt)
        self.nesterov_momentum_coeff = 0.9
        self.opt_m = None

    def apply_gradients(self, grad):
        learning_rate = self.decay_learning_rate()

        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)
        v = self.nesterov_momentum_coeff * self.opt_m - learning_rate * grad
        delta_grad = self.nesterov_momentum_coeff * self.opt_m - (1 + self.nesterov_momentum_coeff) * v
        self.opt_m = v
        LOGGER.debug('In nesterov_momentum, opt_m: {}, v: {}, delta_grad: {}'.format(
            self.opt_m, v, delta_grad
        ))
        return delta_grad


class _AdamOptimizer(_Optimizer):
    def __init__(self, learning_rate, alpha, penalty, decay, decay_sqrt):
        super().__init__(learning_rate, alpha, penalty, decay, decay_sqrt)
        self.opt_beta1 = 0.9
        self.opt_beta2 = 0.999
        self.opt_beta1_decay = 1.0
        self.opt_beta2_decay = 1.0

        self.opt_m = None
        self.opt_v = None

    def apply_gradients(self, grad):
        learning_rate = self.decay_learning_rate()

        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)

        if self.opt_v is None:
            self.opt_v = np.zeros_like(grad)

        self.opt_beta1_decay = self.opt_beta1_decay * self.opt_beta1
        self.opt_beta2_decay = self.opt_beta2_decay * self.opt_beta2
        self.opt_m = self.opt_beta1 * self.opt_m + (1 - self.opt_beta1) * grad
        self.opt_v = self.opt_beta2 * self.opt_v + (1 - self.opt_beta2) * np.square(grad)
        opt_m_hat = self.opt_m / (1 - self.opt_beta1_decay)
        opt_v_hat = self.opt_v / (1 - self.opt_beta2_decay)
        opt_v_hat = np.array(opt_v_hat, dtype=np.float64)
        delta_grad = learning_rate * opt_m_hat / (np.sqrt(opt_v_hat) + 1e-8)
        return delta_grad


class _StochasticQuansiNewtonOptimizer(_Optimizer):
    def __init__(self, learning_rate, alpha, penalty, decay, decay_sqrt):
        super().__init__(learning_rate, alpha, penalty, decay, decay_sqrt)
        self.__opt_hess = None

    def apply_gradients(self, grad):
        learning_rate = self.decay_learning_rate()
        LOGGER.debug("__opt_hess is: {}".format(self.__opt_hess))
        if self.__opt_hess is None:
            delta_grad = learning_rate * grad
        else:
            delta_grad = learning_rate * self.__opt_hess.dot(grad)
            LOGGER.debug("In sqn updater, grad: {}, delta_grad: {}".format(grad, delta_grad))
        return delta_grad

    def set_hess_matrix(self, hess_matrix):
        self.__opt_hess = hess_matrix


def optimizer_factory(param):
    try:
        optimizer_type = param.optimizer
        learning_rate = param.learning_rate
        alpha = param.alpha
        penalty = param.penalty
        decay = param.decay
        decay_sqrt = param.decay_sqrt
        init_params = [learning_rate, alpha, penalty, decay, decay_sqrt]
    except AttributeError:
        raise AttributeError("Optimizer parameters has not been totally set")

    LOGGER.debug("in optimizer_factory, optimizer_type: {}, learning_rate: {}, alpha: {}, penalty: {},"
                 "decay: {}, decay_sqrt: {}".format(optimizer_type, *init_params))

    if optimizer_type == 'sgd':
        return _SgdOptimizer(*init_params)
    elif optimizer_type == 'nesterov_momentum_sgd':
        return _NesterovMomentumSGDOpimizer(*init_params)
    elif optimizer_type == 'rmsprop':
        return _RMSPropOptimizer(*init_params)
    elif optimizer_type == 'adam':
        return _AdamOptimizer(*init_params)
    elif optimizer_type == 'adagrad':
        return _AdaGradOptimizer(*init_params)
    elif optimizer_type == 'sqn':
        return _StochasticQuansiNewtonOptimizer(*init_params)
    else:
        raise NotImplementedError("Optimize method cannot be recognized: {}".format(optimizer_type))