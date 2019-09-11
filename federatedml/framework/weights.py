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

from arch.api.utils.splitable import segment_transfer_enabled
from federatedml.secureprotol.encrypt import Encrypt


class TransferableWeights(metaclass=segment_transfer_enabled()):
    def __init__(self, weights):
        self._weights = weights

    @property
    def unboxed(self):
        return self._weights

    @property
    def weights(self):
        return Weights(self._weights)


class Weights(object):

    def __init__(self, l):
        self._weights = l

    def for_remote(self):
        return TransferableWeights(self._weights)

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
        return self.binary_op(other, operator.add, inplace=False)

    def __truediv__(self, other):
        return self.map_values(lambda x: x / other, inplace=False)

    def __itruediv__(self, other):
        return self.map_values(lambda x: x / other, inplace=True)


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
                self._weights[k] = func(other._weights, v)
            return self
        else:
            _w = dict()
            for k, v in self._weights.items():
                _w[k] = func(other._weights, v)
            return DictWeights(_w)

    def axpy(self, a, y: 'DictWeights'):
        for k, v in self._weights.items():
            self._weights[k] += a * y._weights[k]
