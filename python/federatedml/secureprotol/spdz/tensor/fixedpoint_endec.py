#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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
import functools

import numpy as np

from fate_arch.session import is_table


class FixedPointEndec(object):
    def __init__(self, field: int, base: int, precision_fractional: int, *args, **kwargs):
        self.field = field
        self.base = base
        self.precision_fractional = precision_fractional

    def _encode(self, float_tensor: np.ndarray, check_range=True):
        upscaled = (float_tensor * self.base ** self.precision_fractional).astype(np.int64)
        if check_range:
            assert (np.abs(upscaled) < (self.field / 2)).all(), (
                f"{float_tensor} cannot be correctly embedded: choose bigger field or a lower precision"
            )

        field_element = upscaled % self.field
        return field_element

    def _decode(self, integer_tensor: np.ndarray):
        value = integer_tensor % self.field
        gate = value > self.field // 2
        neg_nums = (value - self.field) * gate
        pos_nums = value * (1 - gate)
        result = (neg_nums + pos_nums) / (self.base ** self.precision_fractional)
        return result

    def _truncate(self, integer_tensor, idx=0):
        if idx == 0:
            return self.field - (self.field - integer_tensor) // (self.base ** self.precision_fractional)
        else:
            return integer_tensor // (self.base ** self.precision_fractional)

    def encode(self, float_tensor, check_range=True):
        if isinstance(float_tensor, (float, np.float)):
            float_tensor = np.array(float_tensor)
        if isinstance(float_tensor, np.ndarray):
            return self._encode(float_tensor, check_range)
        elif is_table(float_tensor):
            f = functools.partial(self._encode, check_range=check_range)
            return float_tensor.mapValues(f)
        else:
            raise ValueError(f"unsupported type: {type(float_tensor)}")

    def decode(self, integer_tensor):
        if isinstance(integer_tensor, (int, np.int16, np.int32, np.int64)):
            integer_tensor = np.array(integer_tensor)
        if isinstance(integer_tensor, np.ndarray):
            return self._decode(integer_tensor)
        elif is_table(integer_tensor):
            f = functools.partial(self._decode)
            return integer_tensor.mapValues(lambda x: f)
        else:
            raise ValueError(f"unsupported type: {type(integer_tensor)}")

    def truncate(self, integer_tensor, idx=0):
        if isinstance(integer_tensor, (int, np.int16, np.int32, np.int64)):
            integer_tensor = np.array(integer_tensor)
        if isinstance(integer_tensor, np.ndarray):
            return self._truncate(integer_tensor, idx)
        elif is_table(integer_tensor):
            f = functools.partial(self._truncate, idx=idx)
            return integer_tensor.mapValues(f)
        else:
            raise ValueError(f"unsupported type: {type(integer_tensor)}")
