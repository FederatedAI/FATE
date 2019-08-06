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
################################################################################
#
#
################################################################################

import numpy as np


class Optimizer(object):
    """  优化模块类；
    """

    def __init__(self, learning_rate, opt_method_name="Sgd"):
        """
        """
        # 优化算法初始化参数；
        self.opt_beta1 = 0.9
        self.opt_beta2 = 0.999
        self.rho = 0.99

        self.opt_beta1_decay = 1.0
        self.opt_beta2_decay = 1.0

        self.opt_m = None
        self.opt_v = None
        self.learning_rate = learning_rate

        self.opt_method_name = opt_method_name.lower()
        self.nesterov_momentum_coeff = 0.9

    def SgdOptimizer(self, grad):
        """sgd 优化；
           grad: 梯度；
        """
        self.learning_rate *= self.opt_beta2
        delta_grad = self.learning_rate * grad
        return delta_grad

    def AdaGradOptimizer(self, grad):
        """ AdaGrad 优化算法；
            grad: 梯度；
        """
        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)

        self.opt_m = self.opt_m + np.square(grad)
        self.opt_m = np.array(self.opt_m, dtype=np.float64)
        delta_grad = self.learning_rate * grad / (np.sqrt(self.opt_m) + 1e-7)
        return delta_grad

    def RMSPropOptimizer(self, grad):
        """ RMSProp 优化算法；
            grad: 梯度；
        """
        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)

        self.opt_m = self.rho * self.opt_m + (1 - self.rho) * np.square(grad)
        self.opt_m = np.array(self.opt_m, dtype=np.float64)
        delta_grad = self.learning_rate * grad / np.sqrt(self.opt_m + 1e-6)
        return delta_grad

    def AdamOptimizer(self, grad):
        """ Adam 优化算法；
            grad: 梯度；
        """
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

    def nesterov_momentum_sgd_opimizer(self, grad):
        if self.opt_m is None:
            self.opt_m = np.zeros_like(grad)
        v = self.nesterov_momentum_coeff * self.opt_m - self.learning_rate * grad
        delta_grad = self.nesterov_momentum_coeff * self.opt_m - (1 + self.nesterov_momentum_coeff) * v
        self.opt_m = v
        # self.learning_rate *= self.nesterov_momentum_coeff_decay
        return delta_grad

    def apply_gradients(self, grad):
        """根据优化器类型应用梯度；
        """
        if self.opt_method_name == "sgd":
            return self.SgdOptimizer(grad)

        elif self.opt_method_name == "rmsprop":
            return self.RMSPropOptimizer(grad)

        elif self.opt_method_name == "adam":
            return self.AdamOptimizer(grad)

        elif self.opt_method_name == "adagrad":
            return self.AdaGradOptimizer(grad)
        elif self.opt_method_name == "momentum":
            return self.nesterov_momentum_sgd_opimizer(grad)

        else:
            raise NotImplementedError("Optimize method cannot be recognized: {}".format(self.opt_method_name))

