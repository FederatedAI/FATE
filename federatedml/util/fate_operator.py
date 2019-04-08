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

    # dot(a, b)[i, j, k, m] = sum(a[i, j, :] * b[k, :, m])
    # One-dimension dot, which is the inner product of these two arrays
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


def get_features_shape(data_instances):
    # LOGGER.debug("In get features shape method, data_instances count: {}".format(
    #     data_instances.count()
    # ))
    if not isinstance(data_instances, types.GeneratorType):
        features = data_instances.collect()
    else:
        features = data_instances

    try:
        one_feature = features.__next__()
    except StopIteration:
        # LOGGER.warning("Data instances is Empty")
        one_feature = None

    if one_feature is not None:
        return one_feature[1].features.shape[0]
    else:
        return None


def get_data_shape(data):
    # LOGGER.debug("In get features shape method, data count: {}".format(
    #     data.count()
    # ))
    if not isinstance(data, types.GeneratorType):
        features = data.collect()
    else:
        features = data

    try:
        one_feature = features.__next__()
    except StopIteration:
        # LOGGER.warning("Data instances is Empty")
        one_feature = None

    if one_feature is not None:
        return len(list(one_feature[1]))
    else:
        return None


def is_empty_feature(data_instances):
    shape_of_feature = get_features_shape(data_instances)
    if shape_of_feature is None or shape_of_feature == 0:
        return True
    return False
