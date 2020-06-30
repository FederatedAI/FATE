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
import array
import functools
import random

import numpy as np

from arch.api.base.table import Table


def rand_tensor(q_field, tensor):
    if isinstance(tensor, Table):
        return tensor.mapValues(
            lambda x: np.random.randint(1, q_field, len(x)).astype(object))
    if isinstance(tensor, np.ndarray):
        arr = np.random.randint(1, q_field, tensor.shape).astype(object)
        return arr
    raise NotImplementedError(f"type={type(tensor)}")


class _MixRand(object):
    def __init__(self, lower, upper, base_size=1000, inc_velocity=0.1, inc_velocity_deceleration=0.01):
        self._lower = lower
        if self._lower < 0:
            raise ValueError(f"lower should great than 0, found {self._lower}")
        self._upper = upper
        if self._upper < self._lower:
            raise ValueError(f"requires upper >= lower, yet upper={upper} and lower={lower}")
        if self._upper <= 0x40000000:
            self._caches = array.array('i')
        else:
            self._caches = array.array('l')

        # generate base random numbers
        for _ in range(base_size):
            self._caches.append(random.SystemRandom().randint(self._lower, self._upper))

        self._inc_rate = inc_velocity
        self._inc_velocity_deceleration = inc_velocity_deceleration

    def _inc(self):
        self._caches.append(random.SystemRandom().randint(self._lower, self._upper))

    def __next__(self):
        if random.random() < self._inc_rate:
            self._inc()
        return self._caches[random.randint(0, len(self._caches) - 1)]

    def __iter__(self):
        return self


def _mix_rand_func(iter, q_field):
    _mix = _MixRand(1, q_field)
    return [(k, np.array([next(_mix) for _ in v], dtype=object)) for k, v in iter]


def urand_tensor(q_field, tensor, use_mix=False):
    if isinstance(tensor, Table):
        if use_mix:
            return tensor.mapPartitions2(functools.partial(_mix_rand_func, q_field=q_field))
        return tensor.mapValues(
            lambda x: np.array([random.SystemRandom().randint(1, q_field) for _ in x], dtype=object))
    if isinstance(tensor, np.ndarray):
        arr = np.zeros(shape=tensor.shape, dtype=object)
        view = arr.view().reshape(-1)
        for i in range(arr.size):
            view[i] = random.SystemRandom().randint(1, q_field)
        return arr
    raise NotImplementedError(f"type={type(tensor)}")
