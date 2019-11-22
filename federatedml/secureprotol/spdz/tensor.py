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

import operator
from typing import Union

import numpy as np

from arch.api.transfer import Party
from federatedml.secureprotol.fate_paillier import PaillierKeypair
from federatedml.secureprotol.spdz import Q_FIELD
from federatedml.secureprotol.spdz import naming
from federatedml.secureprotol.spdz.random_device import RandomDevice
from federatedml.transfer_variable.transfer_class.secret_share_transfer_variable import SecretShareTransferVariable


class SPDZTensorShare(object):
    def __init__(self, value, tensor_name: str = None):
        self.value = value
        self.tensor_name = naming.next_name() if tensor_name is None else tensor_name

    def __str__(self):
        return f"{self.tensor_name}: {self.value}"

    def __repr__(self):
        return self.__str__()

    def rename(self, tensor_name):
        return SPDZTensorShare(value=self.value, tensor_name=tensor_name)

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __matmul__(self, other):
        return SPDZ.get_instance().tensor_dot(self, other, "ij,jk->ik")

    def rescontruct(self):
        return SPDZ.get_instance().rescontruct(self)


def binary(op, x, y, out_name=None):
    if isinstance(x, SPDZTensorShare):
        x_value = x.value
    else:  # assuming x is np.ndarray or a numeric
        x_value = x
    if isinstance(y, SPDZTensorShare):
        y_value = y.value
    else:
        y_value = y
    z_value = op(x_value, y_value)
    return SPDZTensorShare(z_value, out_name)


def add(x, y, out_name=None):
    return binary(operator.add, x, y, out_name)


def sub(x, y, out_name=None):
    return binary(operator.sub, x, y, out_name)


def mul(x, y, out_name=None):
    return binary(operator.mul, x, y, out_name)


def div(x, y, out_name=None):
    return binary(operator.truediv, x, y, out_name)


class SPDZ(object):

    __instance = None

    @classmethod
    def get_instance(cls) -> 'SPDZ':
        return cls.__instance

    @classmethod
    def set_instance(cls, instance):
        cls.__instance = instance

    def __init__(self, name="ss", local_party=None, all_parties=None):

        self._name_service = naming.NamingService(name)
        self._prev_name_service = None

        # define transfer_variables for secret share
        self._transfer_variable = SecretShareTransferVariable()
        self._share_variable = self._transfer_variable.share.disable_auto_clean()
        self._rescontruct_variable = self._transfer_variable.rescontruct.disable_auto_clean()
        self._mul_triplets_encrypted_variable = self._transfer_variable.multiply_triplets_encrypted
        self._mul_triplets_cross_variable = self._transfer_variable.multiply_triplets_cross

        # party list
        self._local_party = self._transfer_variable.local_party() if local_party is None else local_party
        self._all_parties = self._transfer_variable.all_parties() if all_parties is None else all_parties
        self._party_idx = self._all_parties.index(self._local_party)
        self._other_parties = self._all_parties[:self._party_idx] + self._all_parties[(self._party_idx + 1):]

        # paillier's keypair for multiply triplets
        self._public_key, self._private_key = PaillierKeypair.generate_keypair(1024)

        self.set_instance(self)

    def __enter__(self):
        self._prev_name_service = naming.set_naming_service(self._name_service)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        naming.set_naming_service(self._prev_name_service)

    @staticmethod
    def _next_name():
        return naming.next_name()

    def __reduce__(self):
        raise PermissionError("it's unsafe to transfer this")

    def share(self, tensor_name, source: Union[np.ndarray, Party]) -> SPDZTensorShare:
        if isinstance(source, np.ndarray):
            _pre = RandomDevice.rand(source)
            self._remote_share(share=_pre, tensor_name=tensor_name, party=self._other_parties[0])
            for _party in self._other_parties[1:]:
                r = RandomDevice.rand(source)
                self._remote_share(share=r - _pre, tensor_name=tensor_name, party=_party)
                _pre = r
            share = SPDZTensorShare(source - _pre, tensor_name)
        elif isinstance(source, Party):
            share = self._get_share(tensor_name=tensor_name, party=source)[0]
            share = SPDZTensorShare(share, tensor_name)
        else:
            raise ValueError(f"source type {type(source)} unknown")
        return share

    def tensor_dot(self, x, y, einsum_expr, target_name=None):
        if target_name is None:
            target_name = self._next_name()

        def _dot(_x, _y):
            return np.einsum(einsum_expr, _x, _y, optimize=True)

        a, b, c = self.tensor_dot_triplets(x, y, einsum_expr, target_name, schema="rectangle")

        mix_l_share = add(x, a, f"{target_name}_mix_l")
        mix_r_share = add(y, b, f"{target_name}_mix_r")
        mix_l = self.rescontruct(mix_l_share)
        mix_r = self.rescontruct(mix_r_share)

        cross = c - _dot(a, mix_r) - _dot(mix_l, b)
        if self._party_idx == 0:
            cross += _dot(mix_l, mix_r)
        share = SPDZTensorShare(cross, target_name)
        return share

    def tensor_dot_triplets(self, x, y, einsum_expr, transfer_tag=None, schema="rectangle"):
        """
        make shares of tensor dot triplets.

        let a = sum([a_1, ...,a_n]), b = sum([b_1,...,b_n]), c = sum([c_1,...,c_n]), one has
        np.einsum(einsum, a, b) == c

        :param x: first share or tensor or shape
        :param y: second share or tensor or shape
        :param einsum_expr: einstein summation convention, with same notation as that used in nnumpy.einsum
        :param transfer_tag: a tag to identify transfer
        :param schema: triangle or rectangle.
        "triangle":
            $$c_i = T(a_i, b_i) + \\sum_{k=1}^{i-1} (T(a_k, b_i) + T(a_i, b_k) + r_{i, k}) - \\sum_{k=i+1}^n r_{k, i}$$
        "rectangle":
            $$c_i = T(a_i, b_i) + \\sum_{k\neq i} (T(a_i, b_k) + r_{k, i}) - \\sum_{k\neq i} r_{i, k}$$
         where $T(a, b)$ is the einsum of a and b with einsum notation einsum_expr.
        :return: shares a_i, b_i, c_i.
        """
        if transfer_tag is None:
            transfer_tag = self._next_name()

        def _get_shape(inst):
            if isinstance(inst, SPDZTensorShare):
                return inst.value.shape
            if isinstance(inst, np.ndarray):
                return inst.shape
            if isinstance(inst, tuple):
                return inst
            raise ValueError(f"type {type(inst)} unknown")

        a_shape = _get_shape(x)
        b_shape = _get_shape(y)

        random_tensor_a = np.random.randint(Q_FIELD, size=a_shape, dtype=np.int64).astype(object)
        random_tensor_b = np.random.randint(Q_FIELD, size=b_shape, dtype=np.int64).astype(object)

        def _einsum(a, b):
            return np.einsum(einsum_expr, a, b, optimize="optimize")

        c = _einsum(random_tensor_a, random_tensor_b)

        if schema == "triangle":
            def _cross(pair):
                _c = _einsum(pair[0], random_tensor_b) + _einsum(random_tensor_a, pair[1])
                _r = np.random.randint(Q_FIELD, size=_c.shape, dtype=np.int64).astype(object)
                _c += _r
                return _c, _r

            parties_to_send_encrypted = self._all_parties[:self._party_idx]
            parties_to_get_encrypted = self._all_parties[(self._party_idx + 1):]

            # send encrypted pairs to the smaller indexed parties
            if parties_to_send_encrypted:
                encrypt_a = np.vectorize(self._public_key.encrypt)(random_tensor_a)
                encrypt_b = np.vectorize(self._public_key.encrypt)(random_tensor_b)
                self._remote_encrypted_tensor(encrypted=(encrypt_a, encrypt_b),
                                              parties=parties_to_send_encrypted,
                                              tag=transfer_tag)

            # get encrypted pair from the larger indexed parties
            if parties_to_get_encrypted:
                encrypted = self._get_encrypted_tensors(parties=parties_to_get_encrypted,
                                                        tag=transfer_tag)
                for _pair, _p in zip(encrypted, parties_to_get_encrypted):
                    cross, r = _cross(_pair)
                    c -= r
                    self._remote_encrypted_cross_tensor(encrypted=cross,
                                                        parties=_p,
                                                        tag=transfer_tag)
            if parties_to_send_encrypted:
                crosses = self._get_encrypted_cross_tensors(parties=parties_to_send_encrypted,
                                                            tag=transfer_tag)
                for cross in crosses:
                    c += np.vectorize(self._private_key.decrypt, otypes=[object])(cross)

        elif schema == "rectangle":
            def _cross(b):
                _c = _einsum(random_tensor_a, b)
                _r = np.random.randint(Q_FIELD, size=_c.shape, dtype=np.int64).astype(object)
                _c += _r
                return _c, _r

            # encrypt b and broadcast to all other parties
            encrypt_b = np.vectorize(self._public_key.encrypt)(random_tensor_b)
            self._remote_encrypted_tensor(encrypted=encrypt_b,
                                          parties=self._other_parties,
                                          tag=transfer_tag)

            # get encrypted b from all other parties, calc cross terms，then send back
            encrypted_b_list = self._get_encrypted_tensors(parties=self._other_parties,
                                                           tag=transfer_tag)
            for _b, _p in zip(encrypted_b_list, self._other_parties):
                cross, r = _cross(_b)
                c -= r
                self._remote_encrypted_cross_tensor(encrypted=cross,
                                                    parties=_p,
                                                    tag=transfer_tag)

            # get encrypted cross terms, decrypt them
            crosses = self._get_encrypted_cross_tensors(parties=self._other_parties,
                                                        tag=transfer_tag)
            for cross in crosses:
                c += np.vectorize(self._private_key.decrypt, otypes=[object])(cross)

        else:
            raise ValueError(f"schema {schema} unknown")

        return random_tensor_a, random_tensor_b, c

    def rescontruct(self, operand, tensor_name=None):
        if isinstance(operand, SPDZTensorShare):
            share = operand.value
            name = tensor_name or operand.tensor_name
        elif isinstance(operand, np.ndarray):
            share = operand
            name = tensor_name
        else:
            raise ValueError(f"type {type(operand)} unknown")

        if name is None:
            raise ValueError("name not specified")

        # remote share to other parties
        self._broadcast_rescontruct_share(share, name)

        # get shares from other parties
        acc = share
        for other_share in self._get_rescontruct_shares(name):
            acc = acc + other_share
        return acc

    def partial_rescontruct(self):
        # todo: partial parties gets rescontructed tensor
        pass

    def _get_rescontruct_shares(self, tensor_name):
        return self._rescontruct_variable.get_parties(self._other_parties, suffix=(tensor_name,))

    def _broadcast_rescontruct_share(self, share, tensor_name):
        return self._rescontruct_variable.remote_parties(share, self._other_parties, suffix=(tensor_name,))

    def _remote_share(self, share, tensor_name, party):
        return self._share_variable.remote_parties(share, party, suffix=(tensor_name,))

    def _get_share(self, tensor_name, party):
        return self._share_variable.get_parties(party, suffix=(tensor_name,))

    def _remote_encrypted_tensor(self, encrypted, parties, tag):
        return self._mul_triplets_encrypted_variable.remote_parties(encrypted, parties=parties, suffix=tag)

    def _remote_encrypted_cross_tensor(self, encrypted, parties, tag):
        return self._mul_triplets_cross_variable.remote_parties(encrypted, parties=parties, suffix=tag)

    def _get_encrypted_tensors(self, parties, tag):
        return self._mul_triplets_encrypted_variable.get_parties(parties=parties, suffix=tag)

    def _get_encrypted_cross_tensors(self, parties, tag):
        return self._mul_triplets_cross_variable.get_parties(parties=parties, suffix=tag)
