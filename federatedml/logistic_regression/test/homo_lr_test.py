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

from arch.api.utils import log_utils

import unittest

import numpy as np

from federatedml.feature.instance import Instance
from federatedml.optim import Initializer
from federatedml.optim import L2Updater
from federatedml.optim import Optimizer
from federatedml.optim.gradient import LogisticGradient
from federatedml.param.param import InitParam

LOGGER = log_utils.getLogger()


class TestHomoLR(unittest.TestCase):
    def setUp(self):

        self.guest_X = np.array([[1, 2, 3, 4, 5], [3, 2, 4, 5, 1], [2, 2, 3, 1, 1, ]]) / 10
        self.guest_Y = np.array([[1], [1], [-1]])

        self.values = []
        for idx, x in enumerate(self.guest_X):
            inst = Instance(inst_id=idx, features=x, label=self.guest_Y[idx])
            self.values.append((idx, inst))

        self.host_X = np.array([[1, 1.2, 3.1, 4, 5], [2.3, 2, 4, 5.3, 1], [2, 2.2, 1.3, 1, 1.6, ]]) / 10
        self.host_Y = np.array([[-1], [1], [-1]])

        self.host_values = []
        for idx, x in enumerate(self.host_X):
            inst = Instance(inst_id=idx, features=x, label=self.host_Y[idx])
            self.values.append((idx, inst))

        self.max_iter = 10
        self.alpha = 0.01
        self.learning_rate = 0.01
        optimizer = 'SGD'
        self.gradient_operator = LogisticGradient()
        self.initializer = Initializer()
        self.fit_intercept = True
        self.init_param_obj = InitParam(fit_intercept=self.fit_intercept)
        self.updater = L2Updater(self.alpha, self.learning_rate)
        self.optimizer = Optimizer(learning_rate=self.learning_rate, opt_method_name=optimizer)
        self.__init_model()

    def __init_model(self):
        model_shape = self.guest_X.shape[1]
        w = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        if self.fit_intercept:
            self.coef_ = w[:-1]
            self.intercept_ = w[-1]
        else:
            self.coef_ = w
            self.intercept_ = 0
        return w

    def __init_host_model(self):
        model_shape = self.host_X.shape[1]
        w = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        if self.fit_intercept:
            self.host_coef_ = w[:-1]
            self.host_intercept_ = w[-1]
        else:
            self.host_coef_ = w
            self.host_intercept_ = 0
        return w

    def test_one_iter(self):
        w = self.__init_model()
        print("before training, coef: {}, intercept: {}".format(
            self.coef_, self.intercept_
        ))
        self.assertEqual(self.coef_.shape[0], self.guest_X.shape[1])
        grad, loss = self.gradient_operator.compute(self.values, coef=self.coef_,
                    intercept=self.intercept_, fit_intercept=self.fit_intercept)
        loss_norm = self.updater.loss_norm(self.coef_)
        loss = loss + loss_norm
        delta_grad = self.optimizer.apply_gradients(grad)
        self.update_model(delta_grad)
        print("After training, coef: {}, intercept: {}, loss: {}".format(
            self.coef_, self.intercept_, loss
        ))

    def test_multi_iter(self):
        w = self.__init_model()
        loss_hist = [100]
        for iter_num in range(self.max_iter):
            grad, loss = self.gradient_operator.compute(self.values, coef=self.coef_,
                        intercept=self.intercept_, fit_intercept=self.fit_intercept)
            loss_norm = self.updater.loss_norm(self.coef_)
            loss = loss + loss_norm
            delta_grad = self.optimizer.apply_gradients(grad)
            self.update_model(delta_grad)
            self.assertTrue(loss <= loss_hist[-1])
            loss_hist.append(loss)
        print(loss_hist)

    def test_host_iter(self):
        w = self.__init_host_model()
        print("before training, coef: {}, intercept: {}".format(
            self.coef_, self.intercept_
        ))
        self.assertEqual(self.host_coef_.shape[0], self.host_X.shape[1])
        grad, loss = self.gradient_operator.compute(self.host_values,
                                                    coef=self.host_coef_,
                                                    intercept=self.intercept_,
                                                    fit_intercept=self.fit_intercept)
        loss_norm = self.updater.loss_norm(self.coef_)
        # print("***********************************************")
        # print(loss, loss_norm)
        self.assertTrue(loss is None)

    def update_model(self, gradient):
        LOGGER.debug("In update_model function, shape of coef: {}, shape of gradient: {}".format(
            np.shape(self.coef_), np.shape(gradient)
        ))
        if self.fit_intercept:
            if self.updater is not None:
                self.coef_ = self.updater.update_coef(self.coef_, gradient[:-1])
            else:
                self.coef_ = self.coef_ - gradient[:-1]
            self.intercept_ -= gradient[-1]

        else:
            if self.updater is not None:
                self.coef_ = self.updater.update_coef(self.coef_, gradient)
            else:
                self.coef_ = self.coef_ - gradient


if __name__ == '__main__':
    unittest.main()
