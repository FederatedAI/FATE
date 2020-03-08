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

from collections import Iterable

import numpy as np
from scipy.sparse import csr_matrix

from arch.api.utils import log_utils
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber

LOGGER = log_utils.getLogger()


def _one_dimension_dot(X, w):
    res = 0
    # LOGGER.debug("_one_dimension_dot, len of w: {}, len of X: {}".format(len(w), len(X)))
    if isinstance(X, csr_matrix):
        for idx, value in zip(X.indices, X.data):
            res += value * w[idx]
    else:
        for i in range(len(X)):
            if np.fabs(X[i]) < 1e-5:
                continue
            res += w[i] * X[i]

    if res == 0:
        if isinstance(w[0], PaillierEncryptedNumber):
            res = 0 * w[0]
    return res


def dot(value, w):
    if isinstance(value, Instance):
        X = value.features
    else:
        X = value

    # # dot(a, b)[i, j, k, m] = sum(a[i, j, :] * b[k, :, m])
    # # One-dimension dot, which is the inner product of these two arrays

    if np.ndim(X) == np.ndim(w) == 1:
        return _one_dimension_dot(X, w)
    elif np.ndim(X) == 2 and np.ndim(w) == 1:
        res = []
        for x in X:
            res.append(_one_dimension_dot(x, w))
        res = np.array(res)
    else:
        res = np.dot(X, w)

    return res


def vec_dot(x, w):
    new_data = 0
    if isinstance(x, SparseVector):
        for idx, v in x.get_all_data():
            # if idx < len(w):
            new_data += v * w[idx]
    else:
        new_data = np.dot(x, w)
    return new_data


def reduce_add(x, y):
    if x is None and y is None:
        return None

    if x is None:
        return y

    if y is None:
        return x
    if not isinstance(x, Iterable):
        result = x + y
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        result = x + y
    else:
        result = []
        for idx, acc in enumerate(x):
            if acc is None:
                result.append(acc)
                continue
            result.append(acc + y[idx])
    return result


def norm(vector, p=2):
    """
    Get p-norm of this vector

    Parameters
    ----------
    vector : numpy array, Input vector
    p: int, p-norm

    """
    if p < 1:
        raise ValueError('p should larger or equal to 1 in p-norm')

    if type(vector).__name__ != 'ndarray':
        vector = np.array(vector)

    return np.linalg.norm(vector, p)
