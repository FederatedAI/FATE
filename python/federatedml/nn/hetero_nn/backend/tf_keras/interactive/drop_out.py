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
from fate_arch.session import computing_session as session


class DropOut(object):
    def __init__(self, rate, noise_shape):
        self._keep_rate = rate
        self._noise_shape = noise_shape
        self._mask = None
        self._partition = None
        self._mask_table = None
        self._select_mask_table = None
        self._do_backward_select = False

    def forward(self, X):
        forward_x = X * self._mask / self._keep_rate

        return forward_x

    def backward(self, grad):
        if self._do_backward_select:
            self._mask = self._select_mask_table[grad.shape[0]]
            self._select_mask_table = self._select_mask_table[grad.shape[0]:]

        return grad * self._mask / self._keep_rate

    def generate_mask(self):
        self._mask = np.random.uniform(low=0, high=1, size=self._noise_shape) < self._keep_rate

    def generate_mask_table(self):
        _mask_table = session.parallelize(self._mask, include_key=False, partition=self._partition)

        self._mask_table = _mask_table
        return _mask_table

    def set_partition(self, partition):
        self._partition = partition

    def select_backward_sample(self, select_ids):
        select_mask_table = self._mask[np.array(select_ids)]
        if self._select_mask_table is not None:
            self._select_mask_table = np.vstack((self._select_mask_table, select_mask_table))
        else:
            self._select_mask_table = select_mask_table

    def do_backward_select_strategy(self):
        self._do_backward_select = True
