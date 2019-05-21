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

import types
from collections import Iterable

import numpy as np

from federatedml.feature.instance import Instance


def _one_dimension_dot(X, w):
    res = 0
    for i in range(len(X)):
        if np.fabs(X[i]) < 1e-5:
            continue
        res += w[i] * X[i]
    return res


def dot(value, w):
    if isinstance(value, Instance):
        X = value.features
    else:
        X = value

    # # dot(a, b)[i, j, k, m] = sum(a[i, j, :] * b[k, :, m])
    # # One-dimension dot, which is the inner product of these two arrays
    # if np.ndim(X) == np.ndim(w) == 1:
    #     return _one_dimension_dot(X, w)
    # elif np.ndim(X) == 2 and np.ndim(w) == 1:
    #     res = []
    #     for x in X:
    #         res.append(_one_dimension_dot(x, w))
    #     res = np.array(res)
    # else:
    #     res = np.dot(X, w)
    res = np.dot(X, w)
    return res


def reduce_add(x, y):
    if x is None and y is None:
        return None

    if x is None:
        return y

    if y is None:
        return x
    if not isinstance(x, Iterable):
        result = x + y
    else:
        result = []
        for idx, acc in enumerate(x):
            result.append(acc + y[idx])
    return result



