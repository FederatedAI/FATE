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
from federatedml.util import consts
from federatedml.util import fate_operator

from operator import add

LOGGER = log_utils.getLogger()


class LinearGradient(Gradient):
    """
    compute gradient and loss in general setting
    """
    def compute_loss(self, X, Y, coef, intercept):
        """
        compute total loss of a given linear regression model
        :param X: feature matrix
        :param Y: dependent variable
        :param coef: coefficients of given model
        :param intercept: intercept of given model
        :return: total loss value
        """
        tot_loss = np.square(np.add(-Y.transpose(), X.dot(coef) + intercept)).sum()
        return tot_loss

    def compute(self, values, coef, intercept, fit_intercept=True):
        """
        compute gradient and loss of a given linear regression model
        :param values: X, Y matrices
        :param coef: coefficients of given model
        :param intercept: intercept if given model
        :param fit_intercept: boolean, if model has interception or not
        :return: gradient and total loss
        """
        X, Y = self.load_data(values)

        batch_size = len(X)

        if batch_size == 0:
            LOGGER.warning("This partition got 0 data")
            return None, None

        b_gradient = np.add(-Y.transpose(), X.dot(coef) + intercept)
        m_gradient = np.multiply(b_gradient, X)
        if fit_intercept:
            gradient = np.c_[m_gradient, b_gradient]
        else:
            gradient = m_gradient
        grad = sum(gradient)
        loss = self.compute_loss(X, Y, coef, intercept)
        return grad, loss


class HeteroLinearGradient(object):
    """
    Class for compute hetero linear regression gradient and loss
    """
    def __init__(self, encrypt_method=None):
       self.encrypt_operator = encrypt_method

    def compute_wx(self, data_instances, coef_, intercept_=0):
        return data_instances.mapValues(
            lambda v: fate_operator.dot(v.features, coef_) + intercept_)

    @staticmethod
    def __compute_gradient(data, fit_intercept=True):
        """
        Compute hetero-linr gradient for:
        gradient = ∑(wx-y)*x, where residual = (wx-y) has been computed, x is features
        Parameters
        ----------
        data: DTable, include residual and features
        fit_intercept: bool, if hetero-linr has interception or not. Default True

        Returns
        ----------
        numpy.ndarray
            hetero-linr gradient
        """
        feature = []
        residual = []

        for key, value in data:
            feature.append(value[0])
            residual.append(value[1])
        feature = np.array(feature)
        residual = np.array(residual)

        gradient = []
        if feature.shape[0] <= 0:
            return 0
        for j in range(feature.shape[1]):
            feature_col = feature[:, j]
            gradient_j = fate_operator.dot(feature_col, residual)
            gradient.append(gradient_j)

        if fit_intercept:
            bias_grad = np.sum(residual)
            gradient.append(bias_grad)
        gradient.append(feature.shape[0])
        return np.array(gradient)

    def compute_loss(self, data_instances, wx, type):
        """
        Compute hetero-linr loss for:
        Lh = ∑(wx)^2,
        Lg = ∑(wx-y)^2, where y is label, w is model weight and x is features

        Parameters:
        ___________
        param data_instances: DTable, input data
        param wx: DTable, intermediate value
        param type: role type {consts.GUEST, consts.HOST}
        Returns
        ----------
        float
            loss
        """
        if type == consts.HOST:
            loss_square = wx.mapValues(lambda v: np.square(v))
        elif type == consts.GUEST:
            loss_square = wx.join(data_instances, lambda wx, d: wx - int(d.label))
            loss_square = loss_square.mapValues(lambda v: np.square(v))
        else:
            loss_square = 0
            LOGGER.error("Wrong type of role given to compute_loss")
        loss = loss_square.reduce(add)
        return loss

    def compute_residual(self, data_instances, wx, encrypted_wx):
        """
        Compute residual = [[wx_h]] + [[wx_g - y]]
        Parameters
        ----------
        data_instance: DTable, input data
        wx: intermediate value
        encrypted_wx: DTable, encrypted wx

        Returns
        ----------
        DTable
            residual
        """
        d = wx.join(data_instances, lambda wx, d: wx - d.label)
        #LOGGER.debug(list(encrypted_wx.collect()))
        residual = encrypted_wx.join(d, lambda wx,
                                               d: wx + self.encrypt_operator.encrypt(d))
        #LOGGER.debug(list(residual.collect()))
        return residual

    def compute_gradient(self, data_instances, residual, fit_intercept):
        """
        Compute hetero-linr gradient
        Parameters
        ----------
        data_instance: DTable, input data
        residual: DTable, residual =  [[wx_h]] + [[wx_g - y]]
        fit_intercept: bool, if hetero-linr has interception or not

        Returns
        ----------
        DTable
            the hetero-linr's gradient
        """
        #LOGGER.debug(list(residual.collect()))
        feat_join_grad = data_instances.join(residual,
                                            lambda d, g: (d.features, g))
        f = functools.partial(self.__compute_gradient,
                              fit_intercept=fit_intercept)

        gradient_partition = feat_join_grad.mapPartitions(f)
        #LOGGER.debug(list(gradient_partition.collect()))
        gradient_partition = gradient_partition.reduce(
            lambda x, y: x + y)
        #LOGGER.debug(type(gradient_partition))
        gradient = gradient_partition[:-1] / gradient_partition[-1]

        for i in range(len(gradient)):
            if not isinstance(gradient[i], PaillierEncryptedNumber):
                gradient[i] = self.encrypt_operator.encrypt(gradient[i])

        # temporary resource recovery and will be removed in the future
        rubbish_list = [feat_join_grad]
        rubbish_clear(rubbish_list)

        return gradient
