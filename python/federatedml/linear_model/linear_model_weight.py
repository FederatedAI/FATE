#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


import numpy as np

from federatedml.framework.weights import ListWeights, TransferableWeights
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
from federatedml.util import LOGGER


class LinearModelWeights(ListWeights):
    def __init__(self, l, fit_intercept, raise_overflow_error=True):
        l = np.array(l)
        if len(l) > 0 and not isinstance(l[0], PaillierEncryptedNumber):
            if np.max(np.abs(l)) > 1e8:
                if raise_overflow_error:
                    raise RuntimeError("The model weights are overflow, please check if the "
                                       "input data has been normalized")
                else:
                    LOGGER.warning(f"LinearModelWeights contains entry greater than 1e8.")
        super().__init__(l)
        self.fit_intercept = fit_intercept
        self.raise_overflow_error = raise_overflow_error

    def for_remote(self):
        return TransferableWeights(self._weights, self.__class__, self.fit_intercept)

    @property
    def coef_(self):
        if self.fit_intercept:
            return np.array(self._weights[:-1])
        return np.array(self._weights)

    @property
    def intercept_(self):
        if self.fit_intercept:
            return 0.0 if len(self._weights) == 0 else self._weights[-1]
        return 0.0

    def binary_op(self, other: 'LinearModelWeights', func, inplace):
        if inplace:
            for k, v in enumerate(self._weights):
                self._weights[k] = func(self._weights[k], other._weights[k])
            return self
        else:
            _w = []
            for k, v in enumerate(self._weights):
                _w.append(func(self._weights[k], other._weights[k]))
            return LinearModelWeights(_w, self.fit_intercept, self.raise_overflow_error)

    def map_values(self, func, inplace):
        if inplace:
            for k, v in enumerate(self._weights):
                self._weights[k] = func(v)
            return self
        else:
            _w = []
            for v in self._weights:
                _w.append(func(v))
            return LinearModelWeights(_w, self.fit_intercept)

    def __repr__(self):
        return f"weights: {self.coef_}, intercept: {self.intercept_}"
