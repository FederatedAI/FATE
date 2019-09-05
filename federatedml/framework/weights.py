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


class TransferableVariables(metaclass=segment_transfer_enabled()):
    def __init__(self, parameters):
        self.parameters = parameters


class Variables(object):
    def __init__(self, l):
        self._parameter = l

    def for_remote(self):
        return TransferableVariables(self._parameter)

    @staticmethod
    def from_transferable(transferable_parameters: TransferableVariables):
        if isinstance(transferable_parameters.parameters, (list, np.ndarray)):
            return ListVariables(transferable_parameters.parameters)
        if isinstance(transferable_parameters.parameters, dict):
            return DictVariables(transferable_parameters.parameters)

        LOGGER.debug("type of parameters: {}".format(type(transferable_parameters.parameters)))
        raise NotImplemented(f"build parameters from {type(transferable_parameters.parameters)}")

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


class ListVariables(Variables):
    def __init__(self, l):
        super().__init__(l)

    def map_values(self, func, inplace):
        if inplace:
            for k, v in enumerate(self._parameter):
                self._parameter[k] = func(v)
            return self
        else:
            _w = []
            for v in self._parameter:
                _w.append(func(v))
            return ListVariables(_w)

    def binary_op(self, other: 'ListVariables', func, inplace):
        if inplace:
            for k, v in enumerate(self._parameter):
                self._parameter[k] = func(self._parameter[k], other._parameter[k])
            return self
        else:
            _w = []
            for k, v in enumerate(self._parameter):
                _w.append(func(self._parameter[k], other._parameter[k]))
            return ListVariables(_w)

    def axpy(self, a, y: 'ListVariables'):
        for k, v in enumerate(self._parameter):
            self._parameter[k] += a * y._parameter[k]


class DictVariables(Variables):
    def __init__(self, d):
        super().__init__(d)

    def map_values(self, func, inplace):
        if inplace:
            for k, v in self._parameter.items():
                self._parameter[k] = func(v)
            return self
        else:
            _w = dict()
            for k, v in self._parameter.items():
                _w[k] = func(v)
            return DictVariables(_w)

    def binary_op(self, other: 'DictVariables', func, inplace):
        if inplace:
            for k, v in self._parameter.items():
                self._parameter[k] = func(other._parameter, v)
            return self
        else:
            _w = dict()
            for k, v in self._parameter.items():
                _w[k] = func(other._parameter, v)
            return DictVariables(_w)

    def axpy(self, a, y: 'DictVariables'):
        for k, v in self._parameter.items():
            self._parameter[k] += a * y._parameter[k]
