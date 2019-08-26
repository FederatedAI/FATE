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

from federatedml.logistic_regression.logistic_regression_param import LogisticRegressionVariables


class Optimizer(object):
    def __init__(self, learning_rate, alpha, penalty):
        self.learning_rate = learning_rate
        self.iters = 0
        self.alpha = alpha
        self.penalty = penalty

    @property
    def shrinkage_val(self):
        this_step_size = self.learning_rate / np.sqrt(self.iters)
        return self.alpha * this_step_size

    def apply_gradients(self, model_param: LogisticRegressionVariables, grad):
        raise NotImplementedError("Should not call here")

    def l1_updator(self, model_param: LogisticRegressionVariables, gradient):
        coef_ = model_param.coef_
        if model_param.fit_intercept:
            gradient_without_intercept = gradient[: -1]
        else:
            gradient_without_intercept = gradient

        new_weights = np.sign(coef_ - gradient_without_intercept) * \
                      np.maximum(0, np.abs(coef_ - gradient_without_intercept) - self.shrinkage_val)

        if model_param.fit_intercept:
            new_weights = np.append(new_weights, model_param.intercept_)
            new_weights[-1] -= gradient[-1]
        new_param = LogisticRegressionVariables(new_weights, model_param.fit_intercept)
        return new_param

    def l2_updator(self, model_param: LogisticRegressionVariables, gradient):
        coef_ = model_param.coef_
        if model_param.fit_intercept:
            gradient_without_intercept = gradient[: -1]
        else:
            gradient_without_intercept = gradient

        new_weights = coef_ - gradient_without_intercept - self.learning_rate * self.alpha * coef_
        if model_param.fit_intercept:
            new_weights = np.append(new_weights, model_param.intercept_)
            new_weights[-1] -= gradient[-1]
        new_param = LogisticRegressionVariables(new_weights, model_param.fit_intercept)
        return new_param

    def update(self, model_param: LogisticRegressionVariables, grad):
        if self.penalty == 'l1':
            model_param = self.l1_updator(model_param, grad)
        elif self.penalty == 'l2':
            model_param = self.l2_updator(model_param, grad)
        return model_param


class SgdOptimizer(Optimizer):
    def __init__(self, learning_rate, alpha, penalty):
        super().__init__(learning_rate, alpha, penalty)
        self.opt_beta = 0.999

    def apply_gradients(self, model_param: LogisticRegressionVariables, grad):
        self.learning_rate *= self.opt_beta
        new_weights = model_param.parameter - self.learning_rate * grad
        model_param = LogisticRegressionVariables(new_weights, model_param.fit_intercept)
        new_param = self.update(model_param, grad)
        return new_param


class RMSPropOptimizer(Optimizer):
    def __init__(self, learning_rate, alpha, penalty):
        super().__init__(learning_rate, alpha, penalty)
        self.rho = 0.99
        self.opt_m = None

    def apply_gradients(self, model_param: LogisticRegressionVariables, grad):
        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)

        self.opt_m = self.rho * self.opt_m + (1 - self.rho) * np.square(grad)
        self.opt_m = np.array(self.opt_m, dtype=np.float64)
        delta_grad = self.learning_rate * grad / np.sqrt(self.opt_m + 1e-6)
        new_weights = model_param.parameter - delta_grad

        model_param = LogisticRegressionVariables(new_weights, model_param.fit_intercept)
        model_param = self.update(model_param, grad)
        return model_param


class AdaGradOptimizer(Optimizer):
    def __init__(self, learning_rate, alpha, penalty):
        super().__init__(learning_rate, alpha, penalty)
        self.opt_m = None

    def apply_gradients(self, model_param: LogisticRegressionVariables, grad):
        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)
        self.opt_m = self.opt_m + np.square(grad)
        self.opt_m = np.array(self.opt_m, dtype=np.float64)
        delta_grad = self.learning_rate * grad / (np.sqrt(self.opt_m) + 1e-7)
        new_weights = model_param.parameter - delta_grad

        model_param = LogisticRegressionVariables(new_weights, model_param.fit_intercept)
        model_param = self.update(model_param, grad)
        return model_param


class NesterovMomentumSGDOpimizer(Optimizer):
    def __init__(self, learning_rate, alpha, penalty):
        super().__init__(learning_rate, alpha, penalty)
        self.nesterov_momentum_coeff = 0.9
        self.lr_decay = 0.9
        self.opt_m = None

    def apply_gradients(self, model_param: LogisticRegressionVariables, grad):
        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)
        v = self.nesterov_momentum_coeff * self.opt_m - self.learning_rate * grad
        delta_grad = self.nesterov_momentum_coeff * self.opt_m - (1 + self.nesterov_momentum_coeff) * v
        self.opt_m = v
        if self.learning_rate > 0.01:
            self.learning_rate *= self.lr_decay

        new_weights = model_param.parameter - delta_grad

        model_param = LogisticRegressionVariables(new_weights, model_param.fit_intercept)
        model_param = self.update(model_param, grad)
        return model_param


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate, alpha, penalty):
        super().__init__(learning_rate, alpha, penalty)
        self.opt_beta1 = 0.9
        self.opt_beta2 = 0.999
        self.opt_beta1_decay = 1.0
        self.opt_beta2_decay = 1.0

        self.opt_m = None
        self.opt_v = None

    def apply_gradients(self, model_param: LogisticRegressionVariables, grad):
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
        new_weights = model_param.parameter - delta_grad

        model_param = LogisticRegressionVariables(new_weights, model_param.fit_intercept)
        model_param = self.update(model_param, grad)
        return model_param
