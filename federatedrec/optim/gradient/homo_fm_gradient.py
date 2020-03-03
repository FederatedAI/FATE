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

from arch.api.utils import log_utils
from federatedml.util import fate_operator
from federatedml.optim import activation

LOGGER = log_utils.getLogger()


def load_data(data_instance):
    X = []
    Y = []
    for iter_key, instant in data_instance:
        weighted_feature = instant.weight * instant.features
        X.append(weighted_feature)
        if instant.label == 1:
            Y.append([1])
        else:
            Y.append([-1])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def fm_func(features, embed):
    fm = []
    feature_list = list(features)
    for feature in feature_list:
        re = np.multiply(np.expand_dims(feature, 1), embed)
        re = np.sum(re, 0)
        part1 = np.sum(np.power(re, 2))
        features_square = np.power(feature, 2)
        embed_square = np.power(embed, 2)
        part2 = np.sum(np.dot(features_square, embed_square))
        # LOGGER.info("RE after sum:{}part1:{},part2:{}".format(re, part1, part2))
        fm.append(0.5 * (part1 - part2))
    return fm


class FactorizationGradient(object):

    @staticmethod
    def compute_loss(values, w, embed, intercept):
        LOGGER.info("compute loss")
        X, Y = load_data(values)
        fm = fm_func(X, embed)
        func = np.vectorize(activation.log_logistic)
        expo = np.multiply(-Y.transpose(), X.dot(w) + fm + intercept)
        # tot_loss = np.log(1 + np.exp(np.multiply(-Y.transpose(), X.dot(w) + fm + intercept))).sum()
        tot_loss = -func(-expo).sum()
        return tot_loss

    @staticmethod
    def compute_gradient(values, w, embed, intercept, fit_intercept):
        LOGGER.info("compute gradient")
        X, Y = load_data(values)
        batch_size = len(X)
        if batch_size == 0:
            LOGGER.warning("This partition got 0 data")
            return None
        # fm (m,) d (m,1) X (m,n) grad_batch (m,n)
        fm = fm_func(X, embed)
        d = (1.0 / (1 + np.exp(-np.multiply(Y.transpose(), X.dot(w) + fm + intercept))) - 1).transpose() * Y
        grad_batch = d * X
        # m*n*k
        p1 = np.stack([X[i, :][:, np.newaxis] * np.dot(X[i, :], embed)[np.newaxis, :] for i in range(len(X))])
        p2 = np.stack([np.power(X[i, :], 2)[:, np.newaxis] * embed for i in range(len(X))])
        gradvshape = p1.shape
        grad_v = (p1 - p2).reshape(gradvshape[0], -1)
        grad_v = d * grad_v
        # n*k
        # still need to implement the d*grad_v part
        grad_v = sum(grad_v)
        grad_w = sum(grad_batch)
        grad = np.concatenate([grad_w, grad_v.flatten()])
        if fit_intercept:
            grad = np.concatenate([grad, sum(d)])

        return grad
