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

from numpy import random
from arch.api import session
from federatedml.nn.hetero_nn.backend.ops import HeteroNNTensor


class RandomNumberGenerator(object):
    def __init__(self, method, seed=None, loc=0, scale=1):
        random.seed(seed)
        self.generator = getattr(random, method)
        self.loc = loc
        self.scale = scale

    def generate_random_number(self, shape):
        return self.generator(loc=self.loc, scale=self.scale, size=shape)

    def fast_generate_random_number(self, shape, partition=10):
        generator = self.generator
        loc = self.loc
        scale = self.scale

        tb = session.parallelize([None for i in range(shape[0])], include_key=False, partition=partition)

        tb = tb.mapValues(lambda val: generator(loc, scale, shape[1:]))

        return HeteroNNTensor(tb_obj=tb)

