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
from federatedml.optim import activation


class SigmoidBinaryCrossEntropyLoss(object):
    @staticmethod
    def initialize(y):
        return y.mapValues(lambda x: np.zeros(1)), np.zeros(1)

    @staticmethod
    def predict(value):
        return activation.sigmoid(value)

    @staticmethod
    def compute_loss(y, y_prob):
        """
        will remove later
        y: table
        y_prob: table
        """
        logloss = y.join(y_prob, lambda y, yp: (-np.nan_to_num(y * np.log(yp) + (1 - y) * np.log(1 - yp)), 1))
        logloss_sum, sample_num = logloss.reduce(lambda tuple1, tuple2: (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1]))
        return logloss_sum / sample_num

    @staticmethod
    def compute_grad(y, y_pred):
        return y_pred - y

    @staticmethod
    def compute_hess(y, y_pred):
        return y_pred * (1 - y_pred)


class SoftmaxCrossEntropyLoss(object):
    @staticmethod
    def initialize(y, dims=1):
        return y.mapValues(lambda x: np.zeros(dims)), np.zeros(dims)

    @staticmethod
    def predict(values):
        return activation.softmax(values)

    @staticmethod
    def compute_loss(y, y_prob):
        """
        will remove later
        y: table
        y_prob: table
        """
        # np.sum(np.nan_to_num(y_i * np.log(y_pred)), axis=1)
        loss = y.join(y_prob, lambda y, yp_array: (-np.nan_to_num(np.log(yp_array[y])), 1))
        loss_sum, sample_num = loss.reduce(lambda tuple1, tuple2: (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1]))
        return loss_sum / sample_num

    @staticmethod
    def compute_grad(y, y_pred):
        grad = y_pred.copy()
        grad[y] -= 1
        return grad

    @staticmethod
    def compute_hess(y, y_pred):
        return y_pred * (1 - y_pred)
