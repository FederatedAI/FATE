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

import numpy as np
from federatedml.nn.hetero_nn.strategy.comparision import Comparision


class RelativeSelector(object):
    def __init__(self, max_size=None, beta=1, random_state=None, min_prob=0):
        self._comparision = Comparision(size=max_size)
        self._beta = beta
        self._min_prob = min_prob
        np.random.seed(random_state)

    def select_batch_sample(self, samples):
        select_ret = [False] * len(samples)
        for sample in samples:
            self._comparision.add(sample)

        for idx, sample in enumerate(samples):
            select_ret[idx] = max(
                self._min_prob, np.power(
                    np.random.uniform(
                        0, 1), self._beta)) <= self._comparision.get_rate(sample)

        return select_ret


class SelectorFactory(object):
    @staticmethod
    def get_selector(method, selective_size, beta=1, random_rate=None, min_prob=0):
        if not method:
            return None
        elif method == "relative":
            return RelativeSelector(selective_size, beta, random_state=random_rate, min_prob=min_prob)
        else:
            raise ValueError("Back Propagation Selector {} not supported yet")
