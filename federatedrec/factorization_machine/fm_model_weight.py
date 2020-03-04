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

from federatedml.framework.weights import NumpyWeights, TransferableWeights


class FactorizationMachineWeights(NumpyWeights):
    def __init__(self, w, embed, intercept=0., fit_intercept=False):
        if fit_intercept:
            weights = np.concatenate([w.flatten(), embed.flatten(), [intercept]])
        else:
            weights = np.concatenate([w.flatten(), embed.flatten()])
        super().__init__(weights)
        self.w_size = len(w)
        self.embed_shape = embed.shape
        self.fit_intercept = fit_intercept

    def for_remote(self):
        return TransferableWeights(self._weights, self.__class__, self.fit_intercept)

    @property
    def weights(self):
        return self._weights

    @property
    def coef_(self):
        if self.fit_intercept:
            return self._weights[:-1]
        else:
            return self._weights

    @property
    def w_(self):
        return self._weights[:self.w_size]

    @property
    def intercept_(self):
        if self.fit_intercept:
            return self._weights[-1]
        return 0.0

    @property
    def embed_(self):
        if self.fit_intercept:
            return self._weights[self.w_size:-1].reshape(self.embed_shape)
        else:
            return self._weights[self.w_size:].reshape(self.embed_shape)

    def update(self, _weights):
        self._weights = _weights._weights
        return self
