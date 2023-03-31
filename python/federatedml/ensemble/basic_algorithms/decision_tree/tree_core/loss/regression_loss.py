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
import functools
from federatedml.feature.instance import Instance
from federatedml.util import consts
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.loss.loss import Loss


class LeastSquaredErrorLoss(Loss):

    @staticmethod
    def initialize(y):
        y_inst = y.mapValues(lambda label: Instance(features=np.asarray([label])))
        y_inst.schema = {"header": ["label"]}
        statistics = MultivariateStatisticalSummary(y_inst, -1)
        mean = statistics.get_mean()["label"]
        return y.mapValues(lambda x: np.asarray([mean])), np.asarray([mean])

    @staticmethod
    def predict(value):
        return value

    @staticmethod
    def compute_loss(y, y_pred, sample_weight=None):

        lse_loss = y.join(y_pred, lambda y, yp: ((y - yp) * (y - yp), 1))
        avg_loss = Loss.reduce(lse_loss, sample_weights=sample_weight)
        return avg_loss

    @staticmethod
    def compute_grad(y, y_pred):
        return 2 * (y_pred - y)

    @staticmethod
    def compute_hess(y, y_pred):
        if type(y).__name__ == "ndarray" or type(y_pred).__name__ == "ndarray":
            shape = (y - y_pred).shape
            return np.full(shape, 2)
        else:
            return 2


class LeastAbsoluteErrorLoss(Loss):
    @staticmethod
    def initialize(y):
        y_inst = y.mapValues(lambda label: Instance(features=np.asarray([label])))
        y_inst.schema = {"header": ["label"]}
        statistics = MultivariateStatisticalSummary(y_inst, -1)
        median = statistics.get_median()["label"]
        return y.mapValues(lambda x: np.asarray([median])), np.asarray([median])

    @staticmethod
    def predict(value):
        return value

    @staticmethod
    def compute_loss(y, y_pred, sample_weight=None):
        lae_loss = y.join(y_pred, lambda y, yp: (np.abs(y - yp), 1))
        avg_loss = Loss.reduce(lae_loss, sample_weights=sample_weight)
        return avg_loss

    @staticmethod
    def compute_grad(y, y_pred):
        if type(y).__name__ == "ndarray" or type(y_pred).__name__ == "ndarray":
            diff = y_pred - y
            diff[diff > consts.FLOAT_ZERO] = 1
            diff[diff < consts.FLOAT_ZERO] = -1
            diff[np.abs(diff) <= consts.FLOAT_ZERO] = 0
            return diff
        else:
            diff = y_pred - y
            if diff > consts.FLOAT_ZERO:
                return 1
            elif diff < consts.FLOAT_ZERO:
                return -1
            else:
                return 0

    @staticmethod
    def compute_hess(y, y_pred):
        if type(y).__name__ == "ndarray" or type(y_pred).__name__ == "ndarray":
            shape = (y - y_pred).shape
            return np.full(shape, 1)
        else:
            return 1


class HuberLoss(Loss):
    @staticmethod
    def initialize(y):
        y_inst = y.mapValues(lambda label: Instance(features=np.asarray([label])))
        y_inst.schema = {"header": ["label"]}
        statistics = MultivariateStatisticalSummary(y_inst, -1)
        mean = statistics.get_mean()["label"]
        return y.mapValues(lambda x: np.asarray([mean])), np.asarray([mean])

    def __init__(self, delta):
        super().__init__()
        if delta is None:
            self.delta = consts.FLOAT_ZERO
        else:
            self.delta = delta

        if np.abs(self.delta) < consts.FLOAT_ZERO:
            self.delta = consts.FLOAT_ZERO

    def compute_loss(self, y, y_pred, sample_weight=None):
        huber_loss = y.join(y_pred, lambda y, yp:
                            (self.delta ** 2 * (np.sqrt(1 + ((yp - y) / self.delta) ** 2) - 1), 1))
        avg_loss = Loss.reduce(huber_loss, sample_weights=sample_weight)
        return avg_loss

    @staticmethod
    def predict(value):
        return value

    def compute_grad(self, y, y_pred):
        diff = y_pred - y
        return diff / np.sqrt(1.0 + diff * diff / (self.delta ** 2))

    def compute_hess(self, y, y_pred):
        diff = y_pred - y
        return 1.0 / (1.0 + diff * diff / (self.delta ** 2)) ** 1.5


class FairLoss(Loss):
    @staticmethod
    def initialize(y):
        y_inst = y.mapValues(lambda label: Instance(features=np.asarray([label])))
        y_inst.schema = {"header": ["label"]}
        statistics = MultivariateStatisticalSummary(y_inst, -1)
        mean = statistics.get_mean()["label"]
        return y.mapValues(lambda x: np.asarray([mean])), np.asarray([mean])

    def __init__(self, c):
        super().__init__()
        if c is None:
            self.c = consts.FLOAT_ZERO
        else:
            self.c = c

        if np.abs(self.c) < consts.FLOAT_ZERO:
            self.c = consts.FLOAT_ZERO

    @staticmethod
    def predict(value):
        return value

    def compute_loss(self, y, y_pred, sample_weight=None):
        fair_loss = y.join(y_pred, lambda y, yp:
                           (self.c * np.abs(yp - y) - self.c ** 2 * np.log(np.abs(yp - y) / self.c + 1), 1))
        avg_loss = Loss.reduce(fair_loss, sample_weights=sample_weight)
        return avg_loss

    def compute_grad(self, y, y_pred):
        diff = y_pred - y
        return self.c * diff / (np.abs(diff) + self.c)

    def compute_hess(self, y, y_pred):
        diff = y_pred - y
        return self.c ** 2 / (np.abs(diff) + self.c) ** 2


class LogCoshLoss(Loss):

    @staticmethod
    def initialize(y):
        y_inst = y.mapValues(lambda label: Instance(features=np.asarray([label])))
        y_inst.schema = {"header": ["label"]}
        statistics = MultivariateStatisticalSummary(y_inst, -1)
        mean = statistics.get_mean()["label"]
        return y.mapValues(lambda x: np.asarray([mean])), np.asarray([mean])

    @staticmethod
    def predict(value):
        return value

    def compute_loss(self, y, y_pred, sample_weight=None):
        log_cosh_loss = y.join(y_pred, lambda y, yp: (np.log(np.cosh(yp - y)), 1))
        avg_loss = Loss.reduce(log_cosh_loss, sample_weights=sample_weight)
        return avg_loss

    @staticmethod
    def compute_grad(y, y_pred):
        return np.tanh(y_pred - y)

    @staticmethod
    def compute_hess(y, y_pred):
        return 1 - np.tanh(y_pred - y) ** 2


class TweedieLoss(Loss):

    @staticmethod
    def initialize(y):
        # init score = 0, equals to base_score=1.0 in xgb, init_score=log(base_score)=0
        return y.mapValues(lambda x: np.asarray([0])), np.asarray([0])

    def __init__(self, rho=None):
        super().__init__()
        if rho is None:
            self.rho = consts.FLOAT_ZERO
        else:
            self.rho = rho

    @staticmethod
    def predict(value):
        return np.exp(value)

    def compute_loss(self, y, y_pred, sample_weight=None):
        loss_func = functools.partial(self._tweedie_loss, rho=self.rho)
        tweedie_loss = y.join(y_pred, loss_func)
        avg_loss = Loss.reduce(tweedie_loss, sample_weights=sample_weight)
        return avg_loss

    @staticmethod
    def _tweedie_loss(label, pred, rho):
        if pred < 1e-10:
            pred = 1e-10
        a = label * np.exp((1 - rho) * np.log(pred)) / (1 - rho)
        b = np.exp((2 - rho) * np.log(pred)) / (2 - rho)
        return (-a + b), 1

    def compute_grad(self, y, y_pred):
        if y < 0:
            raise ValueError('y < 0, in tweedie loss label must be non-negative, but got {}'.format(y))
        return -y * np.exp((1 - self.rho) * y_pred) + np.exp((2 - self.rho) * y_pred)

    def compute_hess(self, y, y_pred):
        return -y * (1 - self.rho) * np.exp((1 - self.rho) * y_pred) + (2 - self.rho) * np.exp((2 - self.rho) * y_pred)
