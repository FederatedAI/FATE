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


import functools
import math
import random
import sys

import numpy as np
from fate_arch.session import is_table
from federatedml.secureprotol.fixedpoint import FixedPointNumber


FLOAT_MANTISSA_BITS = 32
PRECISION = 2 ** FLOAT_MANTISSA_BITS


def rand_number_generator(q_field):
    number = FixedPointNumber(encoding=random.randint(1, PRECISION),
                              exponent=math.floor((FLOAT_MANTISSA_BITS / 2)
                                                  / FixedPointNumber.LOG2_BASE),
                              n=q_field
                              )

    return number


def rand_tensor(q_field, tensor):
    if is_table(tensor):
        return tensor.mapValues(
            lambda x: np.array([rand_number_generator(q_field=q_field)
                                for _ in x],
                               dtype=FixedPointNumber)
        )
    if isinstance(tensor, np.ndarray):
        arr = np.zeros(shape=tensor.shape, dtype=FixedPointNumber)
        view = arr.view().reshape(-1)
        for i in range(arr.size):
            view[i] = rand_number_generator(q_field=q_field)
        return arr
    raise NotImplementedError(f"type={type(tensor)}")


class _MixRand(object):
    def __init__(self, q_field, base_size=1000, inc_velocity=0.1, inc_velocity_deceleration=0.01):
        self._caches = []
        self._q_field = q_field

        # generate base random numbers
        for _ in range(base_size):
            rand_num = rand_number_generator(q_field=self._q_field)
            self._caches.append(rand_num)

        self._inc_rate = inc_velocity
        self._inc_velocity_deceleration = inc_velocity_deceleration

    def _inc(self):
        rand_num = rand_number_generator(q_field=self._q_field)
        self._caches.append(rand_num)

    def __next__(self):
        if random.random() < self._inc_rate:
            self._inc()
        return self._caches[random.randint(0, len(self._caches) - 1)]

    def __iter__(self):
        return self


def _mix_rand_func(it, q_field):
    _mix = _MixRand(q_field)
    result = []
    for k, v in it:
        result.append((k, np.array([next(_mix) for _ in v], dtype=object)))
    return result


def urand_tensor(q_field, tensor, use_mix=False):
    if is_table(tensor):
        if use_mix:
            return tensor.mapPartitions(functools.partial(_mix_rand_func,
                                                          q_field=q_field),
                                        use_previous_behavior=False,
                                        preserves_partitioning=True)
        return tensor.mapValues(
            lambda x: np.array([rand_number_generator(q_field=q_field)
                                for _ in x],
                               dtype=FixedPointNumber))
    if isinstance(tensor, np.ndarray):
        arr = np.zeros(shape=tensor.shape, dtype=FixedPointNumber)
        view = arr.view().reshape(-1)
        for i in range(arr.size):
            view[i] = rand_number_generator(q_field=q_field)
        return arr
    raise NotImplementedError(f"type={type(tensor)}")
