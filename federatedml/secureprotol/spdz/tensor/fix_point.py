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

from federatedml.secureprotol.spdz.tensor.integer import IntegerTensor


class FixPointEndec(object):

    def __init__(self, field: int, base: int, precision_fractional: int):
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

    def truncate(self, integer_tensor, idx=0):
        if idx == 0:
            return self.field - (self.field - integer_tensor) // (self.base ** self.precision_fractional)
        else:
            return integer_tensor // (self.base ** self.precision_fractional)


class FixPointTensor(IntegerTensor):

    def __init__(self, value, q_field, endec, tensor_name: str = None):
        super().__init__(value, q_field, tensor_name)
        self._endec = endec

    @classmethod
    def from_source(cls, tensor_name, source, **kwargs):
        spdz = cls.get_spdz()
        if 'encoder' in kwargs:
            encoder = kwargs['encoder']
        else:
            base = kwargs['base'] if 'base' in kwargs else 10
            frac = kwargs['frac'] if 'frac' in kwargs else 4
            q_field = kwargs['q_field'] if 'q_field' in kwargs else spdz.q_field
            encoder = FixPointEndec(q_field, base, frac)
        if isinstance(source, np.ndarray):
            source = encoder.encode(source)
        share = cls._share(tensor_name, source)
        return FixPointTensor(share, spdz.q_field, encoder, tensor_name)

    def get(self, tensor_name=None):
        return self._endec.decode(super().get(tensor_name))

    def _boxed(self, value, tensor_name=None):
        return FixPointTensor(value=value, q_field=self.q_field, endec=self._endec, tensor_name=tensor_name)

    def _unboxed(self, other):
        if isinstance(other, FixPointTensor):
            other = other.value
        return other

    def _einstein_summation(self, x, y, einsum_expr, q_field, target_name=None):
        cross = super()._einstein_summation(x, y, einsum_expr, q_field, target_name)
        cross = self._endec.truncate(cross, self.get_spdz().party_idx)
        return cross
