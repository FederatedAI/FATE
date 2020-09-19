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
from federatedml.feature.instance import Instance
from federatedml.util import consts
from federatedml.statistic.statics import MultivariateStatisticalSummary


class LeastSquaredErrorLoss(object):
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
    def compute_loss(y, y_pred):
        lse_loss = y.join(y_pred, lambda y, yp: ((y - yp) * (y - yp), 1))
        lse_sum, sample_num = lse_loss.reduce(lambda tuple1, tuple2: (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1]))
        return lse_sum / sample_num

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


class LeastAbsoluteErrorLoss(object):
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
    def compute_loss(y, y_pred):
        lae = y.join(y_pred, lambda y, yp: (np.abs(y - yp), 1))
        lae_sum, sample_num = lae.reduce(lambda tuple1, tuple2: (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1]))
        return lae_sum / sample_num

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


class HuberLoss(object):
    @staticmethod
    def initialize(y):
        y_inst = y.mapValues(lambda label: Instance(features=np.asarray([label])))
        y_inst.schema = {"header": ["label"]}
        statistics = MultivariateStatisticalSummary(y_inst, -1)
        mean = statistics.get_mean()["label"]
        return y.mapValues(lambda x: np.asarray([mean])), np.asarray([mean])

    def __init__(self, delta):
        if delta is None:
            self.delta = consts.FLOAT_ZERO
        else:
            self.delta = delta

        if np.abs(self.delta) < consts.FLOAT_ZERO:
            self.delta = consts.FLOAT_ZERO

    def compute_loss(self, y, y_pred):
        huber_loss = y.join(y_pred, lambda y, yp:
        (self.delta ** 2 * (np.sqrt(1 + ((yp - y) / self.delta) ** 2) - 1), 1))
        huber_sum, sample_num = huber_loss.reduce(lambda tuple1, tuple2: (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1]))
        return huber_sum / sample_num

    @staticmethod
    def predict(value):
        return value

    def compute_grad(self, y, y_pred):
        diff = y_pred - y
        return diff / np.sqrt(1.0 + diff * diff / (self.delta ** 2))

    def compute_hess(self, y, y_pred):
        diff = y_pred - y
        return 1.0 / (1.0 + diff * diff / (self.delta ** 2)) ** 1.5


class FairLoss(object):
    @staticmethod
    def initialize(y):
        y_inst = y.mapValues(lambda label: Instance(features=np.asarray([label])))
        y_inst.schema = {"header": ["label"]}
        statistics = MultivariateStatisticalSummary(y_inst, -1)
        mean = statistics.get_mean()["label"]
        return y.mapValues(lambda x: np.asarray([mean])), np.asarray([mean])

    def __init__(self, c):
        if c is None:
            self.c = const.FLOAT_ZERO
        else:
            self.c = c

        if np.abs(self.c) < consts.FLOAT_ZERO:
            self.c = consts.FLOAT_ZERO

    @staticmethod
    def predict(value):
        return value

    def compute_loss(self, y, y_pred):
        fair_loss = y.join(y_pred, lambda y, yp:
        (self.c * np.abs(yp - y) - self.c ** 2 * np.log(np.abs(yp - y) / self.c + 1), 1))
        fair_loss_sum, sample_num = fair_loss.reduce(
            lambda tuple1, tuple2: (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1]))
        return fair_loss_sum / sample_num

    def compute_grad(self, y, y_pred):
        diff = y_pred - y
        return self.c * diff / (np.abs(diff) + self.c)

    def compute_hess(self, y, y_pred):
        diff = y_pred - y
        return self.c ** 2 / (np.abs(diff) + self.c) ** 2


class LogCoshLoss(object):
    @staticmethod
    def initialize(y):
        y_inst = y.mapValues(lambda label: Instance(features=np.asarray([label])))
        y_inst.schema = {"header": ["label"]}
        statistics = MultivariateStatisticalSummary(y_inst, -1)
        mean = statistics.get_mean()
        return y.mapValues(lambda x: np.asarray([mean])), np.asarray([mean])

    @staticmethod
    def predict(value):
        return value

    @staticmethod
    def compute_loss(y, y_pred):
        log_cosh_loss = y.join(y_pred, lambda y, yp: (np.log(np.cosh(yp - y)), 1))
        log_cosh_sum, sample_num = log_cosh_loss.reduce(
            lambda tuple1, tuple2: (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1]))
        return log_cosh_sum / sample_num

    @staticmethod
    def compute_grad(y, y_pred):
        return np.tanh(y_pred - y)

    @staticmethod
    def compute_hess(y, y_pred):
        return 1 - np.tanh(y_pred - y) ** 2


class TweedieLoss(object):
    @staticmethod
    def initialize(y):
        y_inst = y.mapValues(lambda label: Instance(features=np.asarray([label])))
        y_inst.schema = {"header": ["label"]}
        statistics = MultivariateStatisticalSummary(y_inst, -1)
        mean = statistics.get_mean()["label"]
        return y.mapValues(lambda x: np.asarray([mean])), np.asarray([mean])

    def __init__(self, rho=None):
        if rho is None:
            self.rho = consts.FLOAT_ZERO
        else:
            self.rho = rho

    @staticmethod
    def predict(value):
        return value

    def compute_loss(self, y, y_pred):
        tweedie_loss = y.join(y_pred,
                              lambda y, yp:
                              (-y * np.exp(1 - self.rho) * np.log(max(yp, consts.FLOAT_ZERO)) / (1 - self.rho) +
                               np.exp(2 - self.rho) * np.log(max(consts.FLOAT_ZERO, yp)) / (2 - self.rho), 1))
        tweedie_loss_sum, sample_num = tweedie_loss.reduce(lambda tuple1, tuple2:
                                                           (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1]))
        return tweedie_loss_sum / sample_num

    def compute_grad(self, y, y_pred):
        return -y * np.exp(1 - self.rho) * y_pred + np.exp(2 - self.rho) * y_pred

    def compute_hess(self, y, y_pred):
        return -y * (1 - self.rho) * np.exp(1 - self.rho) * y_pred + (2 - self.rho) * np.exp(2 - self.rho) * y_pred
