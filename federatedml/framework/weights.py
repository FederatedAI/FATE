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

import abc
import operator

import numpy as np

from arch.api.utils import log_utils
from arch.api.utils.splitable import segment_transfer_enabled
from federatedml.secureprotol.encrypt import Encrypt

LOGGER = log_utils.getLogger()


class TransferableWeights(metaclass=segment_transfer_enabled()):
    def __init__(self, weights, cls, *args, **kwargs):
        self._weights = weights
        self._cls = cls
        if args:
            self._args = args
        if kwargs:
            self._kwargs = kwargs

    def with_degree(self, degree):
        setattr(self, "_degree", degree)
        return self

    def get_degree(self, default=None):
        return getattr(self, "_degree", default)

    @property
    def unboxed(self):
        return self._weights

    @property
    def weights(self):
        if not hasattr(self, "_args") and not hasattr(self, "_kwargs"):
            return self._cls(self._weights)
        else:
            args = self._args if hasattr(self, "_args") else ()
            kwargs = self._kwargs if hasattr(self, "_kwargs") else {}
            return self._cls(self._weights, *args, **kwargs)


class Weights(metaclass=segment_transfer_enabled()):

    def __init__(self, l):
        self._weights = l

    def for_remote(self):
        return TransferableWeights(self._weights, self.__class__)

    @property
    def unboxed(self):
        return self._weights

    @abc.abstractmethod
    def map_values(self, func, inplace):
        pass

    @abc.abstractmethod
    def binary_op(self, other, func, inplace):
        pass

    @abc.abstractmethod
    def axpy(self, a, y):
        pass

    def decrypted(self, cipher: Encrypt, inplace=True):
        return self.map_values(cipher.decrypt, inplace=inplace)

    def encrypted(self, cipher: Encrypt, inplace=True):
        return self.map_values(cipher.encrypt, inplace=inplace)

    def __imul__(self, other):
        return self.map_values(lambda x: x * other, inplace=True)

    def __mul__(self, other):
        return self.map_values(lambda x: x * other, inplace=False)

    def __iadd__(self, other):
        return self.binary_op(other, operator.add, inplace=True)

    def __add__(self, other):
        LOGGER.debug("In binary_op0, _w: {}".format(self._weights))
        return self.binary_op(other, operator.add, inplace=False)

    def __isub__(self, other):
        return self.binary_op(other, operator.sub, inplace=True)

    def __sub__(self, other):
        return self.binary_op(other, operator.sub, inplace=False)

    def __truediv__(self, other):
        return self.map_values(lambda x: x / other, inplace=False)

    def __itruediv__(self, other):
        return self.map_values(lambda x: x / other, inplace=True)


class NumericWeights(Weights):
    def __init__(self, v):
        super().__init__(v)

    def map_values(self, func, inplace):
        v = func(self._weights)
        if inplace:
            self._weights = v
            return self
        else:
            return NumericWeights(v)

    def binary_op(self, other: 'NumpyWeights', func, inplace):
        v = func(self._weights, other._weights)
        if inplace:
            self._weights = v
            return self
        else:
            return NumericWeights(v)

    def axpy(self, a, y: 'NumpyWeights'):
        self._weights = self._weights + a * y._weights
        return self


class ListWeights(Weights):
    def __init__(self, l):
        super().__init__(l)

    def map_values(self, func, inplace):
        if inplace:
            for k, v in enumerate(self._weights):
                self._weights[k] = func(v)
            return self
        else:
            _w = []
            for v in self._weights:
                _w.append(func(v))
            return ListWeights(_w)

    def binary_op(self, other: 'ListWeights', func, inplace):
        if inplace:
            for k, v in enumerate(self._weights):
                self._weights[k] = func(self._weights[k], other._weights[k])
            return self
        else:
            _w = []
            for k, v in enumerate(self._weights):
                _w.append(func(self._weights[k], other._weights[k]))
            return ListWeights(_w)

    def axpy(self, a, y: 'ListWeights'):
        for k, v in enumerate(self._weights):
            self._weights[k] += a * y._weights[k]
        return self


class DictWeights(Weights):

    def __init__(self, d):
        super().__init__(d)

    def map_values(self, func, inplace):
        if inplace:
            for k, v in self._weights.items():
                self._weights[k] = func(v)
            return self
        else:
            _w = dict()
            for k, v in self._weights.items():
                _w[k] = func(v)
            return DictWeights(_w)

    def binary_op(self, other: 'DictWeights', func, inplace):
        if inplace:
            for k, v in self._weights.items():
                self._weights[k] = func(other._weights[k], v)
            return self
        else:
            _w = dict()
            for k, v in self._weights.items():
                _w[k] = func(other._weights[k], v)
            return DictWeights(_w)

    def axpy(self, a, y: 'DictWeights'):
        for k, v in self._weights.items():
            self._weights[k] += a * y._weights[k]
        return self


class OrderDictWeights(Weights):
    """
    This class provide a dict container same as `DictWeights` but with fixed key order.
    This feature is useful in secure aggregation random padding generation, which is order sensitive.
    """

    def __init__(self, d):
        super().__init__(d)
        self.walking_order = sorted(d.keys(), key=str)

    def map_values(self, func, inplace):
        if inplace:
            for k in self.walking_order:
                self._weights[k] = func(self._weights[k])
            return self
        else:
            _w = dict()
            for k in self.walking_order:
                _w[k] = func(self._weights[k])
            return OrderDictWeights(_w)

    def binary_op(self, other: 'OrderDictWeights', func, inplace):
        if inplace:
            for k in self.walking_order:
                self._weights[k] = func(other._weights[k], self._weights[k])
            return self
        else:
            _w = dict()
            for k in self.walking_order:
                _w[k] = func(other._weights[k], self._weights[k])
            return OrderDictWeights(_w)

    def axpy(self, a, y: 'OrderDictWeights'):
        for k in self.walking_order:
            self._weights[k] += a * y._weights[k]
        return self


class NumpyWeights(Weights):
    def __init__(self, arr):
        super().__init__(arr)

    def map_values(self, func, inplace):
        if inplace:
            size = self._weights.size
            view = self._weights.view().reshape(size)
            for i in range(size):
                view[i] = func(view[i])
            return self
        else:
            vec_func = np.vectorize(func)
            weights = vec_func(self._weights)
            return NumpyWeights(weights)

    def binary_op(self, other: 'NumpyWeights', func, inplace):
        if inplace:
            size = self._weights.size
            view = self._weights.view().reshape(size)
            view_other = other._weights.view().reshpae(size)
            for i in range(size):
                view[i] = func(view[i], view_other[i])
            return self
        else:
            vec_func = np.vectorize(func)
            weights = vec_func(self._weights, other._weights)
            return NumpyWeights(weights)

    def axpy(self, a, y: 'NumpyWeights'):
        size = self._weights.size
        view = self._weights.view().reshape(size)
        view_other = y._weights.view().reshpae(size)
        for i in range(size):
            view[i] += a * view_other[i]
        return self
