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

import random
import unittest

import numpy as np
from sklearn import metrics

from arch.api import session
from federatedml.loss import LeastSquaredErrorLoss
from federatedml.loss.regression_loss import LeastAbsoluteErrorLoss
from federatedml.loss.regression_loss import HuberLoss
from federatedml.loss.regression_loss import FairLoss
from federatedml.loss.regression_loss import LogCoshLoss
from federatedml.loss.regression_loss import TweedieLoss
from federatedml.util import consts


class TestLeastSquaredErrorLoss(unittest.TestCase):
    def setUp(self):
        session.init("test_least_squared_error_loss")
        self.lse_loss = LeastSquaredErrorLoss()
        self.y_list = [i % 2 for i in range(100)]
        self.predict_list = [random.random() for i in range(100)]
        self.y = session.parallelize(self.y_list, include_key=False)
        self.predict = session.parallelize(self.predict_list, include_key=False)

    def test_predict(self):
        for y in self.y_list:
            y_pred = self.lse_loss.predict(y)
            self.assertTrue(np.fabs(y_pred - y) < consts.FLOAT_ZERO)

    def test_compute_gradient(self):
        for y, y_pred in zip(self.y_list, self.predict_list):
            lse_grad = self.lse_loss.compute_grad(y, y_pred)
            grad = 2 * (y_pred - y)
            self.assertTrue(np.fabs(lse_grad - grad) < consts.FLOAT_ZERO)

    def test_compute_hess(self):
        for y, y_pred in zip(self.y_list, self.predict_list):
            hess = 2
            lse_hess = self.lse_loss.compute_hess(y, y_pred)
            self.assertTrue(np.fabs(lse_hess - hess) < consts.FLOAT_ZERO)

    def test_compute_loss(self):
        sklearn_loss = metrics.mean_squared_error(self.y_list, self.predict_list)
        lse_loss = self.lse_loss.compute_loss(self.y, self.predict)
        self.assertTrue(np.fabs(lse_loss - sklearn_loss) < consts.FLOAT_ZERO)


class TestLeastAbsoluteErrorLoss(unittest.TestCase):
    def setUp(self):
        session.init("test_least_abs_error_loss")
        self.lae_loss = LeastAbsoluteErrorLoss()
        self.y_list = [i % 2 for i in range(100)]
        self.predict_list = [random.random() for i in range(100)]
        self.y = session.parallelize(self.y_list, include_key=False)
        self.predict = session.parallelize(self.predict_list, include_key=False)

    def test_predict(self):
        for y in self.y_list:
            y_pred = self.lae_loss.predict(y)
            self.assertTrue(np.fabs(y_pred - y) < consts.FLOAT_ZERO)

    def test_compute_gradient(self):
        for y, y_pred in zip(self.y_list, self.predict_list):
            lse_grad = self.lae_loss.compute_grad(y, y_pred)
            diff = y_pred - y
            if diff > consts.FLOAT_ZERO:
                grad = 1
            elif diff < consts.FLOAT_ZERO:
                grad = -1
            else:
                grad = 0

            self.assertTrue(np.fabs(lse_grad - grad) < consts.FLOAT_ZERO)

    def test_compute_hess(self):
        for y, y_pred in zip(self.y_list, self.predict_list):
            hess = 1
            lae_hess = self.lae_loss.compute_hess(y, y_pred)
            self.assertTrue(np.fabs(lae_hess - hess) < consts.FLOAT_ZERO)

    def test_compute_loss(self):
        sklearn_loss = metrics.mean_absolute_error(self.y_list, self.predict_list)
        lae_loss = self.lae_loss.compute_loss(self.y, self.predict)
        self.assertTrue(np.fabs(lae_loss - sklearn_loss) < consts.FLOAT_ZERO)


class TestHuberLoss(unittest.TestCase):
    def setUp(self):
        session.init("test_huber_loss")
        self.delta = 1
        self.huber_loss = HuberLoss(self.delta)
        self.y_list = [i % 2 for i in range(100)]
        self.predict_list = [random.random() for i in range(100)]
        self.y = session.parallelize(self.y_list, include_key=False)
        self.predict = session.parallelize(self.predict_list, include_key=False)

    def test_predict(self):
        for y in self.y_list:
            y_pred = self.huber_loss.predict(y)
            self.assertTrue(np.fabs(y_pred - y) < consts.FLOAT_ZERO)

    def test_compute_gradient(self):
        for y, y_pred in zip(self.y_list, self.predict_list):
            huber_grad = self.huber_loss.compute_grad(y, y_pred)
            diff = y_pred - y
            grad = diff / np.sqrt(diff * diff / self.delta ** 2 + 1)
            self.assertTrue(np.fabs(huber_grad - grad) < consts.FLOAT_ZERO)

    def test_compute_hess(self):
        for y, y_pred in zip(self.y_list, self.predict_list):
            huber_hess = self.huber_loss.compute_hess(y, y_pred)
            diff = y_pred - y
            hess = 1.0 / (1 + diff * diff / self.delta ** 2) ** 1.5
            self.assertTrue(np.fabs(huber_hess - hess) < consts.FLOAT_ZERO)

    def test_compute_loss(self):
        loss = 0
        for y, y_pred in zip(self.y_list, self.predict_list):
            diff = y_pred - y
            loss += self.delta ** 2 * (np.sqrt(1 + diff ** 2 / self.delta ** 2) - 1)
        loss /= len(self.y_list)

        huber_loss = self.huber_loss.compute_loss(self.y, self.predict)
        self.assertTrue(np.fabs(huber_loss - loss) < consts.FLOAT_ZERO)


class TestFairLoss(unittest.TestCase):
    def setUp(self):
        session.init("test_fair_loss")
        self.c = 1
        self.fair_loss = FairLoss(self.c)
        self.y_list = [i % 2 for i in range(100)]
        self.predict_list = [random.random() for i in range(100)]
        self.y = session.parallelize(self.y_list, include_key=False)
        self.predict = session.parallelize(self.predict_list, include_key=False)

    def test_predict(self):
        for y in self.y_list:
            y_pred = self.fair_loss.predict(y)
            self.assertTrue(np.fabs(y_pred - y) < consts.FLOAT_ZERO)

    def test_compute_gradient(self):
        for y, y_pred in zip(self.y_list, self.predict_list):
            fair_grad = self.fair_loss.compute_grad(y, y_pred)
            diff = y_pred - y
            grad = self.c * diff / (np.abs(diff) + self.c)
            self.assertTrue(np.fabs(fair_grad - grad) < consts.FLOAT_ZERO)

    def test_compute_hess(self):
        for y, y_pred in zip(self.y_list, self.predict_list):
            fair_hess = self.fair_loss.compute_hess(y, y_pred)
            diff = y_pred - y
            hess = self.c ** 2 / (np.abs(diff) + self.c) ** 2
            self.assertTrue(np.fabs(fair_hess - hess) < consts.FLOAT_ZERO)

    def test_compute_loss(self):
        loss = 0
        for y, y_pred in zip(self.y_list, self.predict_list):
            diff = y_pred - y
            loss += self.c ** 2 * (np.abs(diff) / self.c - np.log(np.abs(diff) / self.c + 1))
        loss /= len(self.y_list)

        fair_loss = self.fair_loss.compute_loss(self.y, self.predict)
        self.assertTrue(np.fabs(fair_loss - loss) < consts.FLOAT_ZERO)


class TestLogCoshLoss(unittest.TestCase):
    def setUp(self):
        session.init("test_fair_loss")
        self.log_cosh_loss = LogCoshLoss()
        self.y_list = [i % 2 for i in range(100)]
        self.predict_list = [random.random() for i in range(100)]
        self.y = session.parallelize(self.y_list, include_key=False)
        self.predict = session.parallelize(self.predict_list, include_key=False)

    def test_predict(self):
        for y in self.y_list:
            y_pred = self.log_cosh_loss.predict(y)
            self.assertTrue(np.fabs(y_pred - y) < consts.FLOAT_ZERO)

    def test_compute_gradient(self):
        for y, y_pred in zip(self.y_list, self.predict_list):
            log_cosh_grad = self.log_cosh_loss.compute_grad(y, y_pred)
            diff = y_pred - y
            grad = np.tanh(diff)
            self.assertTrue(np.fabs(log_cosh_grad - grad) < consts.FLOAT_ZERO)

    def test_compute_hess(self):
        for y, y_pred in zip(self.y_list, self.predict_list):
            log_cosh_hess = self.log_cosh_loss.compute_hess(y, y_pred)
            diff = y_pred - y
            hess = 1 - np.tanh(diff) ** 2
            self.assertTrue(np.fabs(log_cosh_hess - hess) < consts.FLOAT_ZERO)

    def test_compute_loss(self):
        loss = 0
        for y, y_pred in zip(self.y_list, self.predict_list):
            diff = y_pred - y
            loss += np.log(np.cosh(diff))

        loss /= len(self.y_list)

        log_cosh_loss = self.log_cosh_loss.compute_loss(self.y, self.predict)
        self.assertTrue(np.fabs(log_cosh_loss - loss) < consts.FLOAT_ZERO)


class TestTweedieLoss(unittest.TestCase):
    def setUp(self):
        session.init("test_fair_loss")
        self.rho = 0.5
        self.tweedie_loss = TweedieLoss(self.rho)
        self.y_list = [i % 2 for i in range(100)]
        self.predict_list = [random.random() for i in range(100)]
        self.y = session.parallelize(self.y_list, include_key=False)
        self.predict = session.parallelize(self.predict_list, include_key=False)

    def test_predict(self):
        for y in self.y_list:
            y_pred = self.tweedie_loss.predict(y)
            self.assertTrue(np.fabs(y_pred - y) < consts.FLOAT_ZERO)

    def test_compute_gradient(self):
        for y, y_pred in zip(self.y_list, self.predict_list):
            tweedie_grad = self.tweedie_loss.compute_grad(y, y_pred)
            grad = -y * np.exp(1 - self.rho) * y_pred + np.exp(2 - self.rho) * y_pred
            self.assertTrue(np.fabs(tweedie_grad - grad) < consts.FLOAT_ZERO)

    def test_compute_hess(self):
        for y, y_pred in zip(self.y_list, self.predict_list):
            tweedie_loss_hess = self.tweedie_loss.compute_hess(y, y_pred)
            hess = -y * (1 - self.rho) * np.exp(1 - self.rho) * y_pred + \
                   (2 - self.rho) * np.exp(2 - self.rho) * y_pred

            self.assertTrue(np.fabs(tweedie_loss_hess - hess) < consts.FLOAT_ZERO)

    def test_compute_loss(self):
        loss = 0
        for y, y_pred in zip(self.y_list, self.predict_list):
            if y_pred < consts.FLOAT_ZERO:
                y_pred = consts.FLOAT_ZERO

            a = y * np.exp(1 - self.rho) * np.log(y_pred) / (1 - self.rho)
            b = np.exp(2 - self.rho) * np.log(y_pred) / (2 - self.rho)
            loss += (-a + b)

        loss /= len(self.y_list)

        tweedie_loss = self.tweedie_loss.compute_loss(self.y, self.predict)
        self.assertTrue(np.fabs(tweedie_loss - loss) < consts.FLOAT_ZERO)


if __name__ == "__main__":
    unittest.main()
