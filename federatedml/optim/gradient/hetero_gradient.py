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
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


class HeteroGradientComputer(object):
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
        gradient = ∑(wx-y)*x, where fore_gradient = (wx-y) has been computed, x is features
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

    def compute_gradient(self, data_instances, fore_gradient, fit_intercept):
        """
        Compute hetero-linr gradient
        Parameters
        ----------
        data_instance: DTable, input data
        fore_gradient: DTable, fore_gradient =  [[wx_h]] + [[wx_g]] - y
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

        gradient_partition = feat_join_grad.mapPartitions(f).reduce(lambda x, y: x + y)
        #LOGGER.debug(type(gradient_partition))
        gradient = gradient_partition / data_instances.count()

        return gradient
