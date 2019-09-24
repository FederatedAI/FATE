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

import functools

import numpy as np

from arch.api.utils import log_utils
from federatedml.optim.gradient.base_gradient import Gradient
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
from federatedml.statistic.data_overview import rubbish_clear
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


class LogisticGradient(Gradient):
    def compute_loss(self, values, coef, intercept):
        X, Y = self.load_data(values)
        tot_loss = np.log(1 + np.exp(np.multiply(-Y.transpose(), X.dot(coef) + intercept))).sum()
        return tot_loss

    def compute_gradient(self, values, coef, intercept, fit_intercept):
        X, Y = self.load_data(values)
        batch_size = len(X)
        if batch_size == 0:
            LOGGER.warning("This partition got 0 data")
            return None, None

        d = (1.0 / (1 + np.exp(-np.multiply(Y.transpose(), X.dot(coef) + intercept))) - 1).transpose() * Y
        grad_batch = d * X
        if fit_intercept:
            grad_batch = np.c_[grad_batch, d]
        # grad = sum(grad_batch) / batch_size
        grad = sum(grad_batch)
        return grad


class TaylorLogisticGradient(Gradient):
    def compute_loss(self, values, w, intercept):
        LOGGER.warning("Taylor Logistic Gradient cannot compute loss in encrypted mode")
        return 0

    def compute_gradient(self, values, coef, intercept, fit_intercept):
        X, Y = self.load_data(values)
        batch_size = len(X)
        if batch_size == 0:
            return None

        one_d_y = Y.reshape([-1, ])
        d = (0.25 * np.array(fate_operator.dot(X, coef) + intercept).transpose() + 0.5 * one_d_y * -1)

        grad_batch = X.transpose() * d
        grad_batch = grad_batch.transpose()
        if fit_intercept:
            grad_batch = np.c_[grad_batch, d]
        # grad = sum(grad_batch) / batch_size
        grad = sum(grad_batch)
        return grad
