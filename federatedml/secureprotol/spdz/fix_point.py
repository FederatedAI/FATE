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


class FixPointEndec(object):

    def __init__(self,
                 field: int = 2 ** 62,
                 base: int = 10,
                 precision_fractional: int = 4):
        self.field = field
        self.base = base
        self.precision_fractional = precision_fractional

    def decode(self, integer_tensor: np.ndarray):
        value = integer_tensor % self.field
        gate = value > self.field / 2
        neg_nums = (value - self.field) * gate
        pos_nums = value * (1 - gate)
        result = (neg_nums + pos_nums) / (self.base ** self.precision_fractional)
        return result

    def encode(self, float_tensor: np.ndarray, check_range=True):
        upscaled = (float_tensor * self.base ** self.precision_fractional).astype(np.int64)
        if check_range:
            assert (np.abs(upscaled) < (self.field / 2)).all(), (
                f"{float_tensor} cannot be correctly embedded: choose bigger field or a lower precision"
            )

        field_element = upscaled % self.field
        return field_element

    # def scaler(self, integer_tensor):
    #     value = integer_tensor % self.field
    #     gate = value > self.field / 2
    #     neg_nums = ((value - self.field) / (self.base ** self.precision_fractional) + self.field).astype(int)
    #     pos_nums = (value / (self.base ** self.precision_fractional)).astype(int)
    #     return neg_nums * gate + pos_nums * (1 - gate)

    def scaler2(self, integer_tensor, id=0):
        if id == 0:
            return self.field - (self.field - integer_tensor) // (self.base ** self.precision_fractional)
        else:
            return integer_tensor // (self.base ** self.precision_fractional)

