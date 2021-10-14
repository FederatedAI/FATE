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


import numpy as np

from fate_arch.common import Party
from fate_arch.session import is_table
from federatedml.secureprotol.fixedpoint import FixedPointEndec
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.transfer_variable.transfer_class.secret_share_transfer_variable import SecretShareTransferVariable
from federatedml.util import consts, LOGGER


class SecureMatrix(object):
    # SecureMatrix in SecretSharing With He;
    def __init__(self, party: Party, q_field, other_party):
        self.transfer_variable = SecretShareTransferVariable()
        self.party = party
        self.other_party = other_party
        self.q_field = q_field
        self.encoder = None
        self.get_or_create_endec(self.q_field)

    def get_or_create_endec(self, q_field, **kwargs):
        if self.encoder is None:
            self.encoder = FixedPointEndec(q_field)
        return self.encoder

    @classmethod
    def table_dot(cls, a_table, b_table):
        def _table_dot_func(it):
            ret = None
            for _, (x, y) in it:
                if ret is None:
                    ret = np.tensordot(x, y, [[], []])
                else:
                    ret += np.tensordot(x, y, [[], []])
            return ret

        return a_table.join(b_table, lambda x, y: [x, y]) \
            .applyPartitions(lambda it: _table_dot_func(it)) \
            .reduce(lambda x, y: x + y)

    @classmethod
    def dot(cls, matrix, y):
        def _vec_dot(x, y):
            ret = np.dot(x, y)
            if not isinstance(ret, np.ndarray):
                ret = np.array([ret])
            return ret

        if isinstance(y, np.ndarray):
            ret = matrix.mapValues(lambda x: _vec_dot(x, y))
            return ret

        elif is_table(y):
            ret = cls.table_dot(matrix, y)
            return ret
        else:
            raise ValueError(f"type={type(y)}")

    def secure_matrix_mul(self, matrix, tensor_name, cipher=None, suffix=tuple(), is_fixedpoint_table=True):
        curt_suffix = ("secure_matrix_mul",) + suffix
        dst_role = consts.GUEST if self.party.role == consts.HOST else consts.HOST

        if cipher is not None:
            de_matrix = self.encoder.decode(matrix.value)
            if isinstance(matrix, fixedpoint_table.FixedPointTensor):
                encrypt_mat = cipher.distribute_encrypt(de_matrix)
            else:
                encrypt_mat = cipher.recursive_encrypt(de_matrix)

            # remote encrypted matrix;
            LOGGER.debug(f"In_secure_matrix_mul, encrypt_mat: {encrypt_mat}")
            self.transfer_variable.encrypted_share_matrix.remote(encrypt_mat,
                                                                 role=dst_role,
                                                                 idx=0,
                                                                 suffix=curt_suffix)

            share_tensor = SecureMatrix.from_source(tensor_name,
                                                    self.other_party,
                                                    cipher,
                                                    self.q_field,
                                                    self.encoder,
                                                    is_fixedpoint_table=is_fixedpoint_table)

            return share_tensor

        else:
            share = self.transfer_variable.encrypted_share_matrix.get(role=dst_role,
                                                                      idx=0,
                                                                      suffix=curt_suffix)
            LOGGER.debug(f"Make share tensor")
            if isinstance(matrix, (fixedpoint_table.FixedPointTensor,
                                   fixedpoint_table.PaillierFixedPointTensor)):
                matrix = matrix.value
            ret = SecureMatrix.dot(matrix, share)
            LOGGER.debug(f"tmc, ret: {ret}")
            share_tensor = SecureMatrix.from_source(tensor_name,
                                                    ret,
                                                    cipher,
                                                    self.q_field,
                                                    self.encoder)

            return share_tensor

    @classmethod
    def from_source(cls, tensor_name, source, cipher, q_field, encoder, is_fixedpoint_table=True):
        if is_table(source):
            share_tensor = fixedpoint_table.PaillierFixedPointTensor.from_source(tensor_name=tensor_name,
                                                                                 source=source,
                                                                                 encoder=encoder,
                                                                                 q_field=q_field)
            return share_tensor

        elif isinstance(source, np.ndarray):
            share_tensor = fixedpoint_numpy.PaillierFixedPointTensor.from_source(tensor_name=tensor_name,
                                                                                 source=source,
                                                                                 encoder=encoder,
                                                                                 q_field=q_field)
            return share_tensor

        elif isinstance(source, (fixedpoint_table.PaillierFixedPointTensor,
                                 fixedpoint_numpy.PaillierFixedPointTensor)):
            return cls.from_source(tensor_name, source.value, cipher, q_field, encoder, is_fixedpoint_table)

        elif isinstance(source, Party):
            if is_fixedpoint_table:
                share_tensor = fixedpoint_table.PaillierFixedPointTensor.from_source(tensor_name=tensor_name,
                                                                                     source=source,
                                                                                     encoder=encoder,
                                                                                     q_field=q_field,
                                                                                     cipher=cipher)
            else:
                share_tensor = fixedpoint_numpy.PaillierFixedPointTensor.from_source(tensor_name=tensor_name,
                                                                                     source=source,
                                                                                     encoder=encoder,
                                                                                     q_field=q_field,
                                                                                     cipher=cipher)

            return share_tensor
        else:
            raise ValueError(f"type={type(source)}")
