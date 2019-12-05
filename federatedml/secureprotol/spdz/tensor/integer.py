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

from federatedml.secureprotol.spdz.beaver_triples import beaver_triplets
from federatedml.secureprotol.spdz.utils import NamingService


class IntegerTensor(object):
    __array_ufunc__ = None

    def __init__(self, value, q_field, tensor_name: str = None):
        self.value = value
        self.q_field = q_field
        self.tensor_name = NamingService.get_instance().next() if tensor_name is None else tensor_name

    @classmethod
    def get_spdz(cls):
        from federatedml.secureprotol.spdz import SPDZ
        return SPDZ.get_instance()

    @property
    def shape(self):
        return self.value.shape

    def einsum(self, other, einsum_expr, target_name=None):
        share_val = self._einstein_summation(self, other, einsum_expr, self.q_field, target_name)
        share = self._boxed(share_val)
        return share

    @classmethod
    def from_source(cls, tensor_name, source, **kwargs):
        share = cls._share(tensor_name, source)
        spdz = cls.get_spdz()
        return IntegerTensor(share, spdz.q_field, tensor_name)

    def get(self, tensor_name=None):
        return self._rescontruct(tensor_name)

    def __str__(self):
        return f"{self.tensor_name}: {self.value}"

    def __repr__(self):
        return self.__str__()

    def as_name(self, tensor_name):
        return self._boxed(value=self.value, tensor_name=tensor_name)

    def __add__(self, other):
        z_value = (self.value + self._unboxed(other)) % self.q_field
        return self._boxed(z_value)

    def __radd__(self, other):
        z_value = (self._unboxed(other) + self.value) % self.q_field
        return self._boxed(z_value)

    def __sub__(self, other):
        z_value = (self.value - self._unboxed(other)) % self.q_field
        return self._boxed(z_value)

    def __rsub__(self, other):
        z_value = (self._unboxed(other) - self.value) % self.q_field
        return self._boxed(z_value)

    def __mul__(self, other):
        if not isinstance(other, (int, np.integer)):
            raise NotImplementedError("__mul__ support integer only")
        return self._unboxed(self.value * other)

    def __rmul__(self, other):
        if not isinstance(other, (int, np.integer)):
            raise NotImplementedError("__mul__ support integer only")
        return self._unboxed(self.value * other)

    def __matmul__(self, other):
        return self.einsum(other, "ij,jk->ik")

    @classmethod
    def _share(cls, tensor_name, source):
        spdz = cls.get_spdz()
        if isinstance(source, np.ndarray):
            _pre = spdz.r_device.rand(source.shape)
            spdz.communicator.remote_share(share=_pre, tensor_name=tensor_name, party=spdz.other_parties[0])
            for _party in spdz.other_parties[1:]:
                r = spdz.r_device.rand(source.shape)
                spdz.communicator.remote_share(share=r - _pre, tensor_name=tensor_name, party=_party)
                _pre = r
            return source - _pre

        return spdz.communicator.get_share(tensor_name=tensor_name, party=source)[0]

    def _rescontruct(self, tensor_name=None):
        from federatedml.secureprotol.spdz import SPDZ
        spdz = SPDZ.get_instance()
        share_val = self.value
        name = tensor_name or self.tensor_name

        if name is None:
            raise ValueError("name not specified")

        # remote share to other parties
        spdz.communicator.broadcast_rescontruct_share(share_val, name)

        # get shares from other parties
        for other_share in spdz.communicator.get_rescontruct_shares(name):
            share_val += other_share
            share_val %= self.q_field
        return share_val

    def _einstein_summation(self, x, y, einsum_expr, q_field, target_name=None):
        spdz = self.get_spdz()
        if target_name is None:
            target_name = NamingService.get_instance().next()

        def _einsum(_x, _y):
            return np.einsum(einsum_expr, _x, _y, optimize=True) % q_field

        a, b, c = beaver_triplets(a_shape=x.shape, b_shape=y.shape, einsum_expr=einsum_expr,
                                  q_field=q_field, he_key_pair=(spdz.public_key, spdz.private_key),
                                  communicator=spdz.communicator, name=target_name)

        x_add_a = (x + a)._rescontruct(f"{target_name}_confuse_x")
        y_add_b = (y + b)._rescontruct(f"{target_name}_confuse_y")
        cross = c - _einsum(a, y_add_b) - _einsum(x_add_a, b)
        if spdz.party_idx == 0:
            cross += _einsum(x_add_a, y_add_b)
        cross = cross % q_field
        return cross

    def _unboxed(self, other):
        if isinstance(other, IntegerTensor):
            other = other.value
        return other

    def _boxed(self, value, tensor_name=None):
        return IntegerTensor(value=value, q_field=self.q_field, tensor_name=tensor_name)
