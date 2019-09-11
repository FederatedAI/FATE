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

from math import exp
from operator import add


LOGGER = log_utils.getLogger()


class PoissonGradient(Gradient):
    """
    compute gradient and loss in general setting
    """
    def compute_loss(self, X, Y, coef, intercept):
        """
        compute total loss of a given poisson model
        :param X: feature matrix
        :param Y: dependent variable
        :param coef: coefficients of given model
        :param intercept: intercept of given model
        :return: total loss value
        """
        tot_loss = np.add(-Y.transpose(), np.exp(X.dot(coef) + intercept)).sum()
        return tot_loss
        pass

    def compute(self, values, coef, intercept, fit_intercept=True):
        """
        compute gradient and loss of a given poisson model
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

        b_gradient = np.add(-Y.transpose(), np.exp(X.dot(coef) + intercept))
        m_gradient = np.multiply(b_gradient, X)
        if fit_intercept:
            gradient = np.c_[m_gradient, b_gradient]
        else:
            gradient = m_gradient
        grad = sum(gradient)
        loss = self.compute_loss(X, Y, coef, intercept)
        return grad, loss


class HeteroPoissonGradientComputer(object):
    """
    Class for compute hetero poisson gradient and loss
    """
    def __init__(self, encrypt_method=None):
       self.encrypt_operator = encrypt_method

    def compute_wx(self, data_instances, coef_, intercept_=0):
        return data_instances.mapValues(
                lambda v: np.dot(v.features, coef_) + intercept_)

    def compute_mu(self, data_instances, coef_, intercept_=0, exposure=None):
        if exposure is None:
            mu = data_instances.mapValues(
                lambda v: np.exp(np.dot(v.features, coef_) + intercept_))
        else:
            mu = data_instances.join(exposure,
                                       lambda d, ei: np.exp(np.dot(d.features, coef_) + intercept_) / ei)
        return mu

    def compute_wx_mu(self, data_instances, coef_, intercept_, exposure):
        return data_instances.join(exposure,
                            lambda d, ei: (np.dot(d.features, coef_) + intercept_,
                                           np.exp(np.dot(d.features, coef_) + intercept_) / ei,
                                           d.label,
                                           ei))


    @staticmethod
    def __compute_gradient(data, fit_intercept=True):
        """
        Compute hetero-poisson gradient for:
        gradient = ∑(mu-y)*x, where fore_gradient = (mu-y) has been computed, x is features
        Parameters
        ----------
        data: DTable, include fore_gradient and features
        fit_intercept: bool, if hetero-linr has interception or not. Default True

        Returns
        ----------
        numpy.ndarray
            hetero-linr gradient
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

    def compute_loss(self, guest_forward, host_forward, exposure):
        """
        Compute hetero poisson loss without regular loss:
            loss = sum(exp(mu_g)*exp(mu_h) - y(wx_g + wx_h) - log(exposure))

        Parameters:
        ___________
        param data_instances: DTable, input data
        param mu: DTable, intermediate value
        param wx: DTable, intermediate value
        param en_mu_join_en_wx: DTable, intermediate value from host
        Returns
        ----------
        float
            loss
        """
        loss = guest_forward.join(host_forward, lambda g, h: g[1]*h[1] - g[2] * (g[0] + h[0]))
        loss = loss.join(exposure, lambda l, ei: l + np.log(ei))
        loss = loss.reduce(add) / guest_forward.count()

        return loss


    def compute_fore_gradient(self, guest_forward, host_forward):
        """
        Compute fore_gradient = [[exp(mu_h)]] * exp(mu_g) - y
        Parameters
        ----------
        data_instance: DTable, input data
        mu: intermediate value
        encrypted_mu: DTable, encrypted mu

        Returns
        ----------
        DTable
            fore_gradient
        """
        fore_gradient = guest_forward.join(host_forward, lambda g, h: g[1]*h[1] - g[2])
        #LOGGER.debug(list(fore_gradient.collect()))
        return fore_gradient

    def compute_gradient(self, data_instances, fore_gradient, fit_intercept):
        """
        Compute hetero-linr gradient
        Parameters
        ----------
        data_instance: DTable, input data
        fore_gradient: DTable, fore_gradient =  [[mu_h]] + [[mu_g - y]]
        fit_intercept: bool, if hetero-linr has interception or not

        Returns
        ----------
        DTable
            the hetero-linr's gradient
        """
        #LOGGER.debug(list(fore_gradient.collect()))
        feat_join_grad = data_instances.join(fore_gradient,
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

