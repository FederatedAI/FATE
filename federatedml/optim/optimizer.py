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

from federatedml.logistic_regression.logistic_regression_variables import LogisticRegressionVariables


class _Optimizer(object):
    def __init__(self, learning_rate, alpha, penalty):
        self.learning_rate = learning_rate
        self.iters = 0
        self.alpha = alpha
        self.penalty = penalty

    @property
    def shrinkage_val(self):
        this_step_size = self.learning_rate / np.sqrt(self.iters)
        return self.alpha * this_step_size

    def apply_gradients(self, grad):
        raise NotImplementedError("Should not call here")

    def __l1_updator(self, model_variables: LogisticRegressionVariables, gradient):
        coef_ = model_variables.coef_
        if model_variables.fit_intercept:
            gradient_without_intercept = gradient[: -1]
        else:
            gradient_without_intercept = gradient

        new_weights = np.sign(coef_ - gradient_without_intercept) * \
                      np.maximum(0, np.abs(coef_ - gradient_without_intercept) - self.shrinkage_val)

        if model_variables.fit_intercept:
            new_weights = np.append(new_weights, model_variables.intercept_)
            new_weights[-1] -= gradient[-1]
        new_param = LogisticRegressionVariables(new_weights, model_variables.fit_intercept)
        return new_param

    def __l2_updator(self, model_variables: LogisticRegressionVariables, gradient):
        coef_ = model_variables.coef_
        if model_variables.fit_intercept:
            gradient_without_intercept = gradient[: -1]
        else:
            gradient_without_intercept = gradient

        new_weights = coef_ - gradient_without_intercept - self.learning_rate * self.alpha * coef_
        if model_variables.fit_intercept:
            new_weights = np.append(new_weights, model_variables.intercept_)
            new_weights[-1] -= gradient[-1]
        new_param = LogisticRegressionVariables(new_weights, model_variables.fit_intercept)
        return new_param

    def update(self, model_variables: LogisticRegressionVariables, grad):
        if self.penalty == 'l1':
            model_variables = self.__l1_updator(model_variables, grad)
        elif self.penalty == 'l2':
            model_variables = self.__l2_updator(model_variables, grad)
        else:
            new_vars = model_variables.for_remote().parameters - grad
            model_variables = LogisticRegressionVariables(new_vars, model_variables.fit_intercept)
        return model_variables

    def __l1_loss_norm(self, model_variables: LogisticRegressionVariables):
        coef_ = model_variables.coef_
        loss_norm = np.sum(self.alpha * np.abs(coef_))
        return loss_norm

    def __l2_loss_norm(self, model_variables: LogisticRegressionVariables):
        coef_ = model_variables.coef_
        loss_norm = 0.5 * self.alpha * np.dot(coef_, coef_)
        return loss_norm

    def loss_norm(self, model_variables: LogisticRegressionVariables):
        if self.penalty == 'l1':
            loss_norm_value = self.__l1_loss_norm(model_variables)
        elif self.penalty == 'l2':
            loss_norm_value = self.__l2_loss_norm(model_variables)
        else:
            loss_norm_value = None
        return loss_norm_value

    def update_model(self, model_variables: LogisticRegressionVariables, grad, has_applied=True):
        if not has_applied:
            delta_grad = self.apply_gradients(grad)
        else:
            self.iters += 1
            delta_grad = grad
        new_param = self.update(model_variables, delta_grad)
        return new_param


class _SgdOptimizer(_Optimizer):

    def apply_gradients(self, grad):
        self.iters += 1
        self.learning_rate = self.learning_rate / np.sqrt(self.iters)
        delta_grad = self.learning_rate * grad
        return delta_grad


class _RMSPropOptimizer(_Optimizer):
    def __init__(self, learning_rate, alpha, penalty):
        super().__init__(learning_rate, alpha, penalty)
        self.rho = 0.99
        self.opt_m = None

    def apply_gradients(self, grad):
        self.iters += 1
        self.learning_rate = self.learning_rate / np.sqrt(self.iters)

        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)

        self.opt_m = self.rho * self.opt_m + (1 - self.rho) * np.square(grad)
        self.opt_m = np.array(self.opt_m, dtype=np.float64)
        delta_grad = self.learning_rate * grad / np.sqrt(self.opt_m + 1e-6)
        return delta_grad


class _AdaGradOptimizer(_Optimizer):
    def __init__(self, learning_rate, alpha, penalty):
        super().__init__(learning_rate, alpha, penalty)
        self.opt_m = None

    def apply_gradients(self, grad):
        self.iters += 1
        self.learning_rate = self.learning_rate / np.sqrt(self.iters)

        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)
        self.opt_m = self.opt_m + np.square(grad)
        self.opt_m = np.array(self.opt_m, dtype=np.float64)
        delta_grad = self.learning_rate * grad / (np.sqrt(self.opt_m) + 1e-7)
        return delta_grad


class _NesterovMomentumSGDOpimizer(_Optimizer):
    def __init__(self, learning_rate, alpha, penalty):
        super().__init__(learning_rate, alpha, penalty)
        self.nesterov_momentum_coeff = 0.9
        self.lr_decay = 0.9
        self.opt_m = None

    def apply_gradients(self, grad):
        self.iters += 1
        self.learning_rate = self.learning_rate / np.sqrt(self.iters)

        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)
        v = self.nesterov_momentum_coeff * self.opt_m - self.learning_rate * grad
        delta_grad = self.nesterov_momentum_coeff * self.opt_m - (1 + self.nesterov_momentum_coeff) * v
        self.opt_m = v
        if self.learning_rate > 0.01:
            self.learning_rate *= self.lr_decay
        return delta_grad


class _AdamOptimizer(_Optimizer):
    def __init__(self, learning_rate, alpha, penalty):
        super().__init__(learning_rate, alpha, penalty)
        self.opt_beta1 = 0.9
        self.opt_beta2 = 0.999
        self.opt_beta1_decay = 1.0
        self.opt_beta2_decay = 1.0

        self.opt_m = None
        self.opt_v = None

    def apply_gradients(self, grad):
        self.iters += 1
        self.learning_rate = self.learning_rate / np.sqrt(self.iters)

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
        delta_grad = self.learning_rate * opt_m_hat / (np.sqrt(opt_v_hat) + 1e-8)
        return delta_grad


def optimizer_factory(param):
    try:
        optimizer_type = param.optimizer
        learning_rate = param.learning_rate
        alpha = param.alpha
        penalty = param.penalty
    except AttributeError:
        raise AttributeError("Optimizer parameters has not been totally set")

    if optimizer_type == 'sgd':
        return _SgdOptimizer(learning_rate, alpha, penalty)
    elif optimizer_type == 'nesterov_momentum_sgd':
        return _NesterovMomentumSGDOpimizer(learning_rate, alpha, penalty)
    elif optimizer_type == 'rmsprop':
        return _RMSPropOptimizer(learning_rate, alpha, penalty)
    elif optimizer_type == 'adam':
        return _AdamOptimizer(learning_rate, alpha, penalty)
    elif optimizer_type == 'adagrad':
        return _AdaGradOptimizer(learning_rate, alpha, penalty)
    else:
        raise NotImplementedError("Optimize method cannot be recognized: {}".format(optimizer_type))
