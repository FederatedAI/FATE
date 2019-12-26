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

from arch.api import session
from federatedml.nn.hetero_nn.backend.paillier_tensor import PaillierTensor

BITS = 10


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

    def generate_random_number(self, shape):
        size = self.get_size_by_shape(shape)
        return np.reshape([random.SystemRandom().uniform(self.lower_bound, self.upper_bound) for idx in range(size)],
                          shape)

    def fast_generate_random_number(self, shape, partition=10):
        tb = session.parallelize([None for i in range(shape[0])], include_key=False, partition=partition)

        tb = tb.mapValues(lambda val: self.generate_random_number(shape[1:]))

        return PaillierTensor(tb_obj=tb)
