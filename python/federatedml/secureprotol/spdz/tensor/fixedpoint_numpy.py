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
import functools

import numpy as np

from fate_arch.common import Party
from fate_arch.computing import is_table
from federatedml.secureprotol.spdz.beaver_triples import beaver_triplets
from federatedml.secureprotol.spdz.tensor import fixedpoint_table
from federatedml.secureprotol.spdz.tensor.base import TensorBase
from federatedml.secureprotol.spdz.utils import urand_tensor
# from federatedml.secureprotol.spdz.tensor.fixedpoint_endec import FixedPointEndec
from federatedml.secureprotol.fixedpoint import FixedPointEndec
from federatedml.util import LOGGER


class FixedPointTensor(TensorBase):
    __array_ufunc__ = None

    def __init__(self, value, q_field, endec, tensor_name: str = None):
        super().__init__(q_field, tensor_name)
        self.endec = endec
        self.value = value

    @property
    def shape(self):
        return self.value.shape

    def reshape(self, shape):
        return self._boxed(self.value.reshape(shape))

    def dot(self, other, target_name=None):
        return self.einsum(other, "ij,ik->jk", target_name)

    def dot_local(self, other, target_name=None):
        if isinstance(other, FixedPointTensor):
            other = other.value

        ret = np.dot(self.value, other) % self.q_field
        ret = self.endec.truncate(ret, self.get_spdz().party_idx)

        if not isinstance(ret, np.ndarray):
            ret = np.array([ret])
        return self._boxed(ret, target_name)

    def sub_matrix(self, tensor_name: str, row_indices=None, col_indices=None, rm_row_indices=None,
                   rm_col_indices=None):
        if row_indices is not None:
            x_indices = list(row_indices)
        elif row_indices is None and rm_row_indices is not None:
            x_indices = [i for i in range(self.value.shape[0]) if i not in rm_row_indices]
        else:
            raise RuntimeError(f"invalid argument")

        if col_indices is not None:
            y_indices = list(col_indices)
        elif row_indices is None and rm_col_indices is not None:
            y_indices = [i for i in range(self.value.shape[0]) if i not in rm_col_indices]
        else:
            raise RuntimeError(f"invalid argument")

        value = self.value[x_indices, :][:, y_indices]

        return FixedPointTensor(value=value, q_field=self.q_field, endec=self.endec, tensor_name=tensor_name)

    @classmethod
    def from_source(cls, tensor_name, source, **kwargs):
        spdz = cls.get_spdz()
        q_field = kwargs['q_field'] if 'q_field' in kwargs else spdz.q_field
        if 'encoder' in kwargs:
            encoder = kwargs['encoder']
        else:
            base = kwargs['base'] if 'base' in kwargs else 10
            frac = kwargs['frac'] if 'frac' in kwargs else 4
            encoder = FixedPointEndec(n=q_field, field=q_field, base=base, precision_fractional=frac)
        if isinstance(source, np.ndarray):
            source = encoder.encode(source)
            _pre = urand_tensor(q_field, source)
            spdz.communicator.remote_share(share=_pre, tensor_name=tensor_name, party=spdz.other_parties[0])
            for _party in spdz.other_parties[1:]:
                r = urand_tensor(q_field, source)
                spdz.communicator.remote_share(share=(r - _pre) % q_field, tensor_name=tensor_name, party=_party)
                _pre = r
            share = (source - _pre) % q_field
        elif isinstance(source, Party):
            share = spdz.communicator.get_share(tensor_name=tensor_name, party=source)[0]
        else:
            raise ValueError(f"type={type(source)}")
        return FixedPointTensor(share, q_field, encoder, tensor_name)

    def einsum(self, other: 'FixedPointTensor', einsum_expr, target_name=None):
        spdz = self.get_spdz()
        target_name = target_name or spdz.name_service.next()

        def _dot_func(_x, _y):
            ret = np.dot(_x, _y)
            if not isinstance(ret, np.ndarray):
                ret = np.array([ret])
            return ret
            # return np.einsum(einsum_expr, _x, _y, optimize=True)

        a, b, c = beaver_triplets(a_tensor=self.value, b_tensor=other.value, dot=_dot_func,
                                  q_field=self.q_field, he_key_pair=(spdz.public_key, spdz.private_key),
                                  communicator=spdz.communicator, name=target_name)

        x_add_a = self._raw_add(a).reconstruct(f"{target_name}_confuse_x")
        y_add_b = other._raw_add(b).reconstruct(f"{target_name}_confuse_y")
        cross = c - _dot_func(a, y_add_b) - _dot_func(x_add_a, b)
        if spdz.party_idx == 0:
            cross += _dot_func(x_add_a, y_add_b)
        cross = cross % self.q_field
        cross = self.endec.truncate(cross, self.get_spdz().party_idx)
        share = self._boxed(cross, tensor_name=target_name)
        return share

    def get(self, tensor_name=None, broadcast=True):
        return self.endec.decode(self.reconstruct(tensor_name, broadcast))

    def reconstruct(self, tensor_name=None, broadcast=True):
        from federatedml.secureprotol.spdz import SPDZ
        spdz = SPDZ.get_instance()
        share_val = self.value.copy()
        LOGGER.debug(f"share_val: {share_val}")

        name = tensor_name or self.tensor_name

        if name is None:
            raise ValueError("name not specified")

        # remote share to other parties
        if broadcast:
            spdz.communicator.broadcast_rescontruct_share(share_val, name)

        # get shares from other parties
        for other_share in spdz.communicator.get_rescontruct_shares(name):
            # LOGGER.debug(f"share_val: {share_val}, other_share: {other_share}")
            share_val += other_share
            try:
                share_val %= self.q_field
                return share_val
            except BaseException:
                return share_val

    def transpose(self):
        value = self.value.transpose()
        return self._boxed(value)

    def broadcast_reconstruct_share(self, tensor_name=None):
        from federatedml.secureprotol.spdz import SPDZ
        spdz = SPDZ.get_instance()
        share_val = self.value.copy()
        name = tensor_name or self.tensor_name
        if name is None:
            raise ValueError("name not specified")
        # remote share to other parties
        spdz.communicator.broadcast_rescontruct_share(share_val, name)
        return share_val

    def _boxed(self, value, tensor_name=None):
        return FixedPointTensor(value=value, q_field=self.q_field, endec=self.endec, tensor_name=tensor_name)

    def __str__(self):
        return f"tensor_name={self.tensor_name}, value={self.value}"

    def __repr__(self):
        return self.__str__()

    def as_name(self, tensor_name):
        return self._boxed(value=self.value, tensor_name=tensor_name)

    def _raw_add(self, other):
        z_value = (self.value + other) % self.q_field
        return self._boxed(z_value)

    def _raw_sub(self, other):
        z_value = (self.value - other) % self.q_field
        return self._boxed(z_value)

    def __add__(self, other):
        if isinstance(other, PaillierFixedPointTensor):
            z_value = (self.value + other)
            return PaillierFixedPointTensor(z_value)
        elif isinstance(other, FixedPointTensor):
            return self._raw_add(other.value)
        z_value = (self.value + other) % self.q_field
        return self._boxed(z_value)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, PaillierFixedPointTensor):
            z_value = (self.value - other)
            return PaillierFixedPointTensor(z_value)
        elif isinstance(other, FixedPointTensor):
            return self._raw_sub(other.value)
        z_value = (self.value - other) % self.q_field
        return self._boxed(z_value)

    def __rsub__(self, other):
        if isinstance(other, (PaillierFixedPointTensor, FixedPointTensor)):
            return other - self
        z_value = (other - self.value) % self.q_field
        return self._boxed(z_value)

    def __mul__(self, other):
        if isinstance(other, PaillierFixedPointTensor):
            z_value = self.value * other.value
            return PaillierFixedPointTensor(z_value)

        if isinstance(other, FixedPointTensor):
            other = other.value

        z_value = self.value * other
        z_value = z_value % self.q_field
        z_value = self.endec.truncate(z_value, self.get_spdz().party_idx)

        return self._boxed(z_value)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        return self.einsum(other, "ij,jk->ik")


class PaillierFixedPointTensor(TensorBase):
    __array_ufunc__ = None

    def __init__(self, value, tensor_name: str = None, cipher=None):
        super().__init__(q_field=None, tensor_name=tensor_name)
        self.value = value
        self.cipher = cipher

    def dot(self, other, target_name=None):
        def _vec_dot(x, y):
            ret = np.dot(x, y)
            if not isinstance(ret, np.ndarray):
                ret = np.array([ret])
            return ret

        if isinstance(other, (FixedPointTensor, fixedpoint_table.FixedPointTensor)):
            other = other.value
        if isinstance(other, np.ndarray):
            ret = _vec_dot(self.value, other)
            return self._boxed(ret, target_name)
        elif is_table(other):
            f = functools.partial(_vec_dot,
                                  self.value)
            ret = other.mapValues(f)
            return fixedpoint_table.PaillierFixedPointTensor(value=ret,
                                                             tensor_name=target_name,
                                                             cipher=self.cipher)
        else:
            raise ValueError(f"type={type(other)}")

    def broadcast_reconstruct_share(self, tensor_name=None):
        from federatedml.secureprotol.spdz import SPDZ
        spdz = SPDZ.get_instance()
        share_val = self.value.copy()
        name = tensor_name or self.tensor_name
        if name is None:
            raise ValueError("name not specified")
        # remote share to other parties
        spdz.communicator.broadcast_rescontruct_share(share_val, name)
        return share_val

    def __str__(self):
        return f"tensor_name={self.tensor_name}, value={self.value}"

    def __repr__(self):
        return self.__str__()

    def _raw_add(self, other):
        z_value = (self.value + other)
        return self._boxed(z_value)

    def _raw_sub(self, other):
        z_value = (self.value - other)
        return self._boxed(z_value)

    def __add__(self, other):
        if isinstance(other, (PaillierFixedPointTensor, FixedPointTensor)):
            return self._raw_add(other.value)
        else:
            return self._raw_add(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (PaillierFixedPointTensor, FixedPointTensor)):
            return self._raw_sub(other.value)
        else:
            return self._raw_sub(other)

    def __rsub__(self, other):
        if isinstance(other, (PaillierFixedPointTensor, FixedPointTensor)):
            z_value = other.value - self.value
        else:
            z_value = other - self.value
        return self._boxed(z_value)

    def __mul__(self, other):
        if isinstance(other, PaillierFixedPointTensor):
            raise NotImplementedError("__mul__ not support PaillierFixedPointTensor")
        elif isinstance(other, FixedPointTensor):
            return self._boxed(self.value * other.value)
        else:
            return self._boxed(self.value * other)

    def __rmul__(self, other):
        self.__mul__(other)

    def _boxed(self, value, tensor_name=None):
        return PaillierFixedPointTensor(value=value,
                                        tensor_name=tensor_name,
                                        cipher=self.cipher)

    @classmethod
    def from_source(cls, tensor_name, source, **kwargs):
        spdz = cls.get_spdz()
        q_field = kwargs['q_field'] if 'q_field' in kwargs else spdz.q_field
        if 'encoder' in kwargs:
            encoder = kwargs['encoder']
        else:
            base = kwargs['base'] if 'base' in kwargs else 10
            frac = kwargs['frac'] if 'frac' in kwargs else 4
            encoder = FixedPointEndec(n=q_field, field=q_field, base=base, precision_fractional=frac)

        if isinstance(source, np.ndarray):
            _pre = urand_tensor(q_field, source)

            share = _pre

            spdz.communicator.remote_share(share=source - encoder.decode(_pre),
                                           tensor_name=tensor_name,
                                           party=spdz.other_parties[-1])

            return FixedPointTensor(value=share,
                                    q_field=q_field,
                                    endec=encoder,
                                    tensor_name=tensor_name)

        elif isinstance(source, Party):
            share = spdz.communicator.get_share(tensor_name=tensor_name, party=source)[0]

            is_cipher_source = kwargs['is_cipher_source'] if 'is_cipher_source' in kwargs else True
            if is_cipher_source:
                cipher = kwargs['cipher']
                share = cipher.recursive_decrypt(share)
                share = encoder.encode(share)
            return FixedPointTensor(value=share,
                                    q_field=q_field,
                                    endec=encoder,
                                    tensor_name=tensor_name)
        else:
            raise ValueError(f"type={type(source)}")
