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

LOGGER = log_utils.getLogger()


def hard_sigmoid(x):
    y = 0.2 * x + 0.5
    return np.clip(y, 0, 1)


def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)


def sigmoid(x):
    if x <= 0:
        a = np.exp(x)
        a /= (1. + a)
    else:
        a = 1. / (1. + np.exp(-x))
    return a


def softplus(x):
    return np.log(1. + np.exp(x))


def softsign(x):
    return x / (1 + np.abs(x))


def tanh(x):
    return np.tanh(x)


def log_logistic(x):
    if x <= 0:
        a = x - np.log(1 + np.exp(x))
    else:
        a = - np.log(1 + np.exp(-x))
    return a
