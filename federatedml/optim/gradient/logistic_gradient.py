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
    def compute_loss(self, X, Y, coef, intercept):
        tot_loss = np.log(1 + np.exp(np.multiply(-Y.transpose(), X.dot(coef) + intercept))).sum()
        return tot_loss

    def compute(self, values, coef, intercept, fit_intercept):

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
    """
    Class for compute hetero-lr gradient and loss
    """

    def __init__(self, encrypt_method=None):
        """
        Parameters
        ----------
        encrypt_obj: Object, encrypt object set in hetero-lr, like Paillier, it should be inited before.
        """
        self.encrypt_operator = encrypt_method

    @staticmethod
    def __compute_gradient(data, fit_intercept=True):
        """
        Compute hetero-lr gradient for:
        gradient = ∑(1/2*ywx-1)*1/2yx, where fore_gradient = (1/2*ywx-1)*1/2y has been computed, x is features
        Parameters
        ----------
        data: DTable, include fore_gradient and features
        fit_intercept: bool, if hetero-lr has interception or not. Default True

        Returns
        ----------
        numpy.ndarray
            hetero-lr gradient
        """
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
        """
        Compute hetero-lr loss for:
        loss = log2 - 1/2*ywx + 1/8*(wx)^2, where y is label, w is model weight and x is features
        Parameters
        ----------
        values: DTable, include 1/2*ywx and (wx)^2

        numpy.ndarray
            hetero-lr loss
        """
        bias = np.log(2)

        loss = 0
        counter = 0
        for _, value in enumerate(values):
            l = value[0] * (-1) + value[1] / 8 + bias
            loss = loss + l

            counter += 1

        return np.array([loss, counter])

    def compute_fore_gradient(self, data_instance, encrypted_wx):
        """
        Compute fore_gradient = (1/2*ywx-1)*1/2y
        Parameters
        ----------
        data_instance: DTable, input data
        encrypted_wx: DTable, encrypted wx

        Returns
        ----------
        DTable
            fore_gradient
        """
        fore_gradient = encrypted_wx.join(data_instance, lambda wx, d: 0.25 * wx - 0.5 * d.label)
        return fore_gradient

    def compute_gradient(self, data_instance, fore_gradient, fit_intercept):
        """
        Compute hetero-lr gradient
        Parameters
        ----------
        data_instance: DTable, input data
        fore_gradient: DTable, fore_gradient = (1/2*ywx-1)*1/2y
        fit_intercept: bool, if hetero-lr has interception or not

        Returns
        ----------
        DTable
            the hetero-lr's gradient
        """
        feat_join_grad = data_instance.join(fore_gradient, lambda d, g: (d.features, g))
        f = functools.partial(self.__compute_gradient, fit_intercept=fit_intercept)

        gradient_partition = feat_join_grad.mapPartitions(f).reduce(lambda x, y: x + y)
        gradient = gradient_partition[:-1] / gradient_partition[-1]

        for i in range(len(gradient)):
            if not isinstance(gradient[i], PaillierEncryptedNumber):
                gradient[i] = self.encrypt_operator.encrypt(gradient[i])

        # temporary resource recovery and will be removed in the future
        rubbish_list = [feat_join_grad]
        rubbish_clear(rubbish_list)

        return gradient

    def compute_gradient_and_loss(self, data_instance, fore_gradient, encrypted_wx, en_sum_wx_square, fit_intercept):
        """
        Compute gradient and loss
        Parameters
        ----------
        data_instance: DTable, input data
        fore_gradient: DTable, fore_gradient = (1/2*ywx-1)*1/2y
        encrypted_wx: DTable, encrypted wx
        en_sum_wx_square: DTable, encrypted wx^2
        fit_intercept: bool, if hetero-lr has interception or not

        Returns
        ----------
        DTable
            the hetero-lr gradient and loss
        """
        # compute gradient
        gradient = self.compute_gradient(data_instance, fore_gradient, fit_intercept)

        # compute and loss
        half_ywx = encrypted_wx.join(data_instance, lambda wx, d: 0.5 * wx * int(d.label))
        half_ywx_join_en_sum_wx_square = half_ywx.join(en_sum_wx_square, lambda yz, ez: (yz, ez))
        f = functools.partial(self.__compute_loss)
        loss_partition = half_ywx_join_en_sum_wx_square.mapPartitions(f).reduce(lambda x, y: x + y)
        loss = loss_partition[0] / loss_partition[1]

        # temporary resource recovery and will be removed in the future
        rubbish_list = [half_ywx, half_ywx_join_en_sum_wx_square]
        rubbish_clear(rubbish_list)

        return gradient, loss
