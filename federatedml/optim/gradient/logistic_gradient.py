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
from federatedml.optim.federated_aggregator import HeteroFederatedAggregator
from federatedml.optim.gradient.base_gradient import Gradient
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


class LogisticGradient(Gradient):
    def compute_loss(self, X, Y, coef, intercept):
        tot_loss = np.log(1 + np.exp(np.multiply(-Y.transpose(), X.dot(coef) + intercept))).sum()
        # avg_loss = tot_loss / Y.shape[0]
        # avg_loss = LogLoss.compute(X, Y, coef)
        return tot_loss

    def compute(self, values, coef, intercept, fit_intercept):

        # LOGGER.debug("In logistic gradient compute method")
        # print("In logistic gradient compute method")
        X, Y = self.load_data(values)

        # print("Data loaded, shape of X : {}, shape of Y: {}, coef shape: {}".format(
        #     X.shape, Y.shape, np.shape(coef)))
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
        loss = self.compute_loss(X, Y, coef, intercept)
        return grad, loss


class TaylorLogisticGradient(Gradient):
    def compute_loss(self, X, Y, w, intercept):
        LOGGER.warning("Taylor Logistic Gradient cannot compute loss in encrypted mode")
        return 0

    def compute(self, values, coef, intercept, fit_intercept):
        X, Y = self.load_data(values)
        batch_size = len(X)
        if batch_size == 0:
            return None, None

        one_d_y = Y.reshape([-1, ])
        d = (0.25 * np.array(fate_operator.dot(X, coef) + intercept).transpose() + 0.5 * one_d_y * -1)

        grad_batch = X.transpose() * d
        grad_batch = grad_batch.transpose()
        if fit_intercept:
            grad_batch = np.c_[grad_batch, d]
        # grad = sum(grad_batch) / batch_size
        grad = sum(grad_batch)
        return grad, None


class HeteroLogisticGradient(object):
    def __init__(self, encrypt_method=None):
        self.encrypt_operator = encrypt_method

    @staticmethod
    def __compute_gradient(data, fit_intercept=True):
        feature = []
        fore_gradient = []

        for key, value in data:
            feature.append(value[0])
            fore_gradient.append(value[1])
        feature = np.array(feature)
        fore_gradient = np.array(fore_gradient)

        gradient = []
        if feature.shape[0] <= 0:
            return 0
        for j in range(feature.shape[1]):
            feature_col = feature[:, j]
            gradient_j = fate_operator.dot(feature_col, fore_gradient)
            gradient.append(gradient_j)

        if fit_intercept:
            bias_grad = np.sum(fore_gradient)
            gradient.append(bias_grad)
        gradient.append(feature.shape[0])
        return np.array(gradient)

    @staticmethod
    def __compute_loss(values):
        half_ywx = []
        encrypted_wx_square = []
        bias = np.log(2)

        for key, value in values:
            half_ywx.append(value[0])
            encrypted_wx_square.append(value[1])

        if len(half_ywx) <= 0 or len(encrypted_wx_square) <= 0:
            return 0

        loss = 0
        for i in range(len(half_ywx)):
            l = half_ywx[i] * (-1) + encrypted_wx_square[i] / 8 + bias
            if i == 0:
                loss = l
            else:
                loss = loss + l

        return np.array([loss, len(half_ywx)])

    def compute_fore_gradient(self, data_instance, encrypted_wx):
        fore_gradient = encrypted_wx.join(data_instance, lambda wx, d: 0.25 * wx - 0.5 * d.label)
        return fore_gradient

    def compute_gradient(self, data_instance, fore_gradient, fit_intercept):
        feat_join_grad = data_instance.join(fore_gradient, lambda d, g: (d.features, g))
        f = functools.partial(self.__compute_gradient, fit_intercept=fit_intercept)

        gradient_partition = feat_join_grad.mapPartitions(f).reduce(lambda x, y: x + y)
        gradient = gradient_partition[:-1] / gradient_partition[-1]

        for i in range(len(gradient)):
            if not isinstance(gradient[i], PaillierEncryptedNumber):
                gradient[i] = self.encrypt_operator.encrypt(gradient[i])


        return gradient

    def compute_gradient_and_loss(self, data_instance, fore_gradient, encrypted_wx, en_sum_wx_square, fit_intercept):
        # compute gradient
        gradient = self.compute_gradient(data_instance, fore_gradient, fit_intercept)

        # compute and loss
        half_ywx = encrypted_wx.join(data_instance, lambda wx, d: 0.5 * wx * int(d.label))
        half_ywx_join_en_sum_wx_square = half_ywx.join(en_sum_wx_square, lambda yz, ez: (yz, ez))
        f = functools.partial(self.__compute_loss)
        loss_partition = half_ywx_join_en_sum_wx_square.mapPartitions(f).reduce(lambda x, y: x + y)
        loss = loss_partition[0] / loss_partition[1]

        return gradient, loss
