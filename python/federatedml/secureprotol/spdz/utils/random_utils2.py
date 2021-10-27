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
import random

import numpy as np
from fate_arch.session import is_table
from federatedml.secureprotol.fixedpoint import FixedPointNumber


def rand_tensor(q_field, tensor):
    precision = 2 ** 64
    if is_table(tensor):
        return tensor.mapValues(
            lambda x: np.array([FixedPointNumber(encoding=random.randint(1, precision),
                                                 exponent=FixedPointNumber.calculate_exponent_from_precision(
                                                     precision=precision),
                                                 n=q_field
                                                 )
                                for _ in x],
                               dtype=FixedPointNumber)
        )
    if isinstance(tensor, np.ndarray):
        arr = np.zeros(shape=tensor.shape, dtype=FixedPointNumber)
        view = arr.view().reshape(-1)
        for i in range(arr.size):
            view[i] = FixedPointNumber(encoding=random.randint(1, precision),
                                       exponent=FixedPointNumber.calculate_exponent_from_precision(precision=precision),
                                       n=q_field
                                       )
        return arr
    raise NotImplementedError(f"type={type(tensor)}")


class _MixRand(object):
    def __init__(self, lower, upper, q_field, base_size=1000, inc_velocity=0.1, inc_velocity_deceleration=0.01):
        self._lower = lower
        if self._lower < 0:
            raise ValueError(f"lower should great than 0, found {self._lower}")
        self._upper = upper
        if self._upper < self._lower:
            raise ValueError(f"requires upper >= lower, yet upper={upper} and lower={lower}")

        self._q_field = q_field
        if self._q_field < self._upper:
            raise ValueError(f"requires q_field >= upper, yet upper={q_field} and lower={upper}")

        self._caches = []

        # generate base random numbers
        for _ in range(base_size):
            rand_num = FixedPointNumber(encoding=random.SystemRandom().randint(self._lower, self._upper),
                                        exponent=FixedPointNumber.calculate_exponent_from_precision(
                                            precision=self._upper),
                                        n=self._q_field)
            self._caches.append(rand_num)

        self._inc_rate = inc_velocity
        self._inc_velocity_deceleration = inc_velocity_deceleration

    def _inc(self):
        rand_num = FixedPointNumber(encoding=random.SystemRandom().randint(self._lower, self._upper),
                                    exponent=FixedPointNumber.calculate_exponent_from_precision(
                                        precision=self._upper),
                                    n=self._q_field)
        self._caches.append(rand_num)

    def __next__(self):
        if random.random() < self._inc_rate:
            self._inc()
        return self._caches[random.randint(0, len(self._caches) - 1)]

    def __iter__(self):
        return self


def _mix_rand_func(it, precision, q_field):
    _mix = _MixRand(1, precision, q_field)
    result = []
    for k, v in it:
        result.append((k, np.array([next(_mix) for _ in v], dtype=object)))
    return result


def urand_tensor(q_field, tensor, use_mix=False):
    precision = 2 ** 64
    if is_table(tensor):
        if use_mix:
            return tensor.mapPartitions(functools.partial(_mix_rand_func,
                                                          precision=precision,
                                                          q_field=q_field),
                                        use_previous_behavior=False,
                                        preserves_partitioning=True)
        return tensor.mapValues(
            lambda x: np.array([FixedPointNumber(encoding=random.SystemRandom().randint(1, precision),
                                                 exponent=FixedPointNumber.calculate_exponent_from_precision(
                                                     precision=precision),
                                                 n=q_field
                                                 )
                                for _ in x],
                               dtype=FixedPointNumber))
    if isinstance(tensor, np.ndarray):
        arr = np.zeros(shape=tensor.shape, dtype=FixedPointNumber)
        view = arr.view().reshape(-1)
        for i in range(arr.size):
            view[i] = FixedPointNumber(encoding=random.SystemRandom().randint(1, precision),
                                       exponent=FixedPointNumber.calculate_exponent_from_precision(precision=precision),
                                       n=q_field
                                       )
        return arr
    raise NotImplementedError(f"type={type(tensor)}")
