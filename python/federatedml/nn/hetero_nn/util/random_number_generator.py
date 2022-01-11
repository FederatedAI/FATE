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
#

import random

import numpy as np
from fate_arch.session import computing_session

from federatedml.secureprotol.paillier_tensor import PaillierTensor

BITS = 10
MIXED_RATE = 0.5


class RandomNumberGenerator(object):
    def __init__(self):
        self.lower_bound = -2 ** BITS
        self.upper_bound = 2 ** BITS

    @staticmethod
    def get_size_by_shape(shape):
        size = 1
        for dim in shape:
            size *= dim

        return size

    def generate_random_number_1d(self, size, mixed_rate=MIXED_RATE, keep=None):
        if keep is not None:
            ret = [0] * size
            for i in range(size):
                if keep[i]:
                    rng = random.SystemRandom().uniform(
                        self.lower_bound, self.upper_bound) if np.random.rand() < mixed_rate else np.random.uniform(
                        self.lower_bound, self.upper_bound)
                    ret[i] = rng

            return np.array(ret)[keep]
        else:
            return [
                random.SystemRandom().uniform(
                    self.lower_bound,
                    self.upper_bound) if np.random.rand() < mixed_rate else np.random.uniform(
                    self.lower_bound,
                    self.upper_bound) for _ in range(size)]

    def generate_random_number(self, shape=None, mixed_rate=MIXED_RATE, keep=None):
        if keep is not None:
            size = self.get_size_by_shape(keep.shape)
            return self.generate_random_number_1d(size, mixed_rate=mixed_rate, keep=keep)
        else:
            size = self.get_size_by_shape(shape)
            return np.reshape(self.generate_random_number_1d(size, mixed_rate=mixed_rate), shape)

    def fast_generate_random_number(self, shape, partition=10, mixed_rate=MIXED_RATE, keep_table=None):
        if keep_table:
            tb = keep_table.mapValues(lambda keep_array: self.generate_random_number(keep=keep_array,
                                                                                     mixed_rate=mixed_rate))
            return PaillierTensor(tb)
        else:
            tb = computing_session.parallelize([None for _ in range(shape[0])], include_key=False, partition=partition)

            tb = tb.mapValues(lambda val: self.generate_random_number(shape[1:], mixed_rate=mixed_rate))

            return PaillierTensor(tb)
