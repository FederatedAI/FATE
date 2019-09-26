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
from federatedml.loss import SigmoidBinaryCrossEntropyLoss
from federatedml.loss import SoftmaxCrossEntropyLoss
from federatedml.util import consts


class TestSigmoidBinaryCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        session.init("test_cross_entropy")
        self.sigmoid_loss = SigmoidBinaryCrossEntropyLoss()
        self.y_list = [i % 2 for i in range(100)]
        self.predict_list = [random.random() for i in range(100)]
        self.y = session.parallelize(self.y_list, include_key=False)
        self.predict = session.parallelize(self.predict_list, include_key=False)

    def test_predict(self):
        for i in range(1, 10):
            np_v = 1.0 / (1.0 + np.exp(-1.0 / i))
            self.assertTrue(np.fabs(self.sigmoid_loss.predict(1.0 / i) - np_v) < consts.FLOAT_ZERO)

    def test_compute_gradient(self):
        for i in range(10):
            pred = random.random()
            y = i % 2
            grad = pred - y
            self.assertTrue(np.fabs(self.sigmoid_loss.compute_grad(y, pred) - grad) < consts.FLOAT_ZERO)

    def test_compute_hess(self):
        for i in range(10):
            pred = random.random()
            y = i % 2
            hess = pred * (1 - pred)
            self.assertTrue(np.fabs(self.sigmoid_loss.compute_hess(y, pred) - hess) < consts.FLOAT_ZERO)

    def test_compute_loss(self):
        sklearn_loss = metrics.log_loss(self.y_list, self.predict_list)
        sigmoid_loss = self.sigmoid_loss.compute_loss(self.y, self.predict)
        self.assertTrue(np.fabs(sigmoid_loss - sklearn_loss) < consts.FLOAT_ZERO)


class TestSoftmaxCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        session.init("test_cross_entropy")
        self.softmax_loss = SoftmaxCrossEntropyLoss()
        self.y_list = [i % 5 for i in range(100)]
        self.predict_list = [np.array([random.random() for i in range(5)]) for j in range(100)]
        self.y = session.parallelize(self.y_list, include_key=False)
        self.predict = session.parallelize(self.predict_list, include_key=False)

    def test_predict(self):
        for i in range(10):
            list = [random.random() for j in range(5)]
            pred_arr = np.asarray(list, dtype='float64')
            mx = pred_arr.max()
            predict = np.exp(pred_arr - mx) / sum(np.exp(pred_arr - mx))
            softmaxloss_predict = self.softmax_loss.predict(pred_arr)
            self.assertTrue(np.fabs(predict - softmaxloss_predict).all() < consts.FLOAT_ZERO)

    def test_compute_grad(self):
        for i in range(10):
            pred = np.asarray([random.random() for j in range(5)], dtype="float64")
            label = random.randint(0, 4)
            softmaxloss_grad = self.softmax_loss.compute_grad(label, pred)
            grad = pred.copy()
            grad[label] -= 1
            self.assertTrue(np.fabs(grad - softmaxloss_grad).all() < consts.FLOAT_ZERO)

    def test_compute_hess(self):
        for i in range(10):
            pred = np.asarray([random.random() for j in range(5)], dtype='float64')
            label = random.randint(0, 4)
            softmaxloss_hess = self.softmax_loss.compute_hess(label, pred)
            hess = pred * (1 - pred)
            self.assertTrue(np.fabs(hess - softmaxloss_hess).all() < consts.FLOAT_ZERO)

    def test_compute_loss(self):
        softmax_loss = self.softmax_loss.compute_loss(self.y, self.predict)
        loss = sum(-np.log(pred[yi]) for yi, pred in zip(self.y_list, self.predict_list)) / len(self.y_list)
        self.assertTrue(np.fabs(softmax_loss - loss) < consts.FLOAT_ZERO)


if __name__ == '__main__':
    unittest.main()
