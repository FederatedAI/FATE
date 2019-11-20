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

import numpy as np

from arch.api.utils import log_utils
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


def load_data(data_instance):
    X = []
    Y = []
    for iter_key, instant in data_instance:
        weighted_feature = instant.weight * instant.features
        X.append(weighted_feature)
        if instant.label == 1:
            Y.append([1])
        else:
            Y.append([-1])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


class LogisticGradient(object):

    @staticmethod
    def compute_loss(values, coef, intercept):
        X, Y = load_data(values)
        tot_loss = np.log(1 + np.exp(np.multiply(-Y.transpose(), X.dot(coef) + intercept))).sum()
        return tot_loss

    @staticmethod
    def compute_gradient(values, coef, intercept, fit_intercept):
        X, Y = load_data(values)
        batch_size = len(X)
        if batch_size == 0:
            LOGGER.warning("This partition got 0 data")
            return None

        d = (1.0 / (1 + np.exp(-np.multiply(Y.transpose(), X.dot(coef) + intercept))) - 1).transpose() * Y
        grad_batch = d * X
        if fit_intercept:
            grad_batch = np.c_[grad_batch, d]
        grad = sum(grad_batch)
        return grad


class TaylorLogisticGradient(object):

    @staticmethod
    def compute_gradient(values, coef, intercept, fit_intercept):
        LOGGER.debug("Get in compute_gradient")
        X, Y = load_data(values)
        batch_size = len(X)
        if batch_size == 0:
            return None

        one_d_y = Y.reshape([-1, ])
        d = (0.25 * np.array(fate_operator.dot(X, coef) + intercept).transpose() + 0.5 * one_d_y * -1)

        grad_batch = X.transpose() * d
        grad_batch = grad_batch.transpose()
        if fit_intercept:
            grad_batch = np.c_[grad_batch, d]
        grad = sum(grad_batch)
        LOGGER.debug("Finish compute_gradient")
        return grad
