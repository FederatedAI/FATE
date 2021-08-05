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
import operator
from abc import ABC

import numpy as np

from fate_arch import session
from federatedml.framework.hetero.procedure import batch_generator
from federatedml.linear_model.linear_model_base import BaseLinearModel
from federatedml.secureprotol.fixedpoint import FixedPointEndec
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.transfer_variable.transfer_class.batch_generator_transfer_variable import \
    BatchGeneratorTransferVariable
from federatedml.transfer_variable.transfer_class.sshe_model_transfer_variable import SSHEModelTransferVariable
from federatedml.transfer_variable.transfer_class.converge_checker_transfer_variable import \
    ConvergeCheckerTransferVariable
from federatedml.util import consts, LOGGER


class SSHEModelBase(BaseLinearModel, ABC):
    def __init__(self):
        super().__init__()
        self._set_parties()
        self.transfer_variable = SSHEModelTransferVariable()
        self.fixpoint_encoder = None
        self.random_field = 2 << 20
        self.encrypted_source_features = None
        self.converge_transfer_variable = ConvergeCheckerTransferVariable()

    def _init_model(self, params):
        super(SSHEModelBase, self)._init_model(params)
        self.batch_generator = batch_generator.Guest() if self.role == consts.GUEST else batch_generator.Host()
        LOGGER.debug(f"batch_generator: {self.batch_generator}, self.role: {self.role}")
        self.batch_generator.register_batch_generator(BatchGeneratorTransferVariable(), has_arbiter=False)

    def check_converge_by_weights(self, last_w, new_w, suffix):
        raise NotImplementedError("Should not call here")

    def _set_parties(self):
        parties = []
        guest_parties = session.get_latest_opened().parties.roles_to_parties(["guest"])
        host_parties = session.get_latest_opened().parties.roles_to_parties(["host"])
        # if len(guest_parties) != 1 or len(host_parties) != 1:
        #     raise ValueError(
        #         f"one guest and one host required, "
        #         f"while {len(guest_parties)} guest and {len(host_parties)} host provided"
        #     )
        parties.extend(guest_parties)
        parties.extend(host_parties)

        local_party = session.get_latest_opened().parties.local_party
        other_party = parties[0] if parties[0] != local_party else parties[1]

        self.parties = parties
        self.local_party = local_party
        self.other_party = other_party

    @staticmethod
    def create_fixpoint_encoder(n, **kwargs):
        # base = kwargs['base'] if 'base' in kwargs else 10
        # frac = kwargs['frac'] if 'frac' in kwargs else 4
        # q_field = kwargs['q_field'] if 'q_field' in kwargs else spdz.q_field
        # encoder = fixedpoint_numpy.FixedPointObjectEndec(q_field, base, frac)
        encoder = FixedPointEndec(n)
        return encoder

    def share_matrix(self, matrix_tensor, suffix=tuple()):
        curt_suffix = ("share_matrix",) + suffix
        table = matrix_tensor.value
        r = fixedpoint_table.urand_tensor(q_field=self.random_field,
                                          tensor=table)
        r = self.fixpoint_encoder.encode(r)
        # LOGGER.debug(f"In_share_matrix, r: {r.first()}")
        if isinstance(matrix_tensor, fixedpoint_table.FixedPointTensor):
            random_tensor = fixedpoint_table.FixedPointTensor.from_value(value=r,
                                                                         encoder=matrix_tensor.endec,
                                                                         q_field=self.random_field)
            to_share = matrix_tensor.value.join(random_tensor.value, operator.sub)
        elif isinstance(matrix_tensor, fixedpoint_numpy.FixedPointTensor):
            random_tensor = fixedpoint_numpy.FixedPointTensor.from_value(value=r,
                                                                         encoder=matrix_tensor.endec,
                                                                         q_field=self.random_field)
            to_share = (matrix_tensor - random_tensor).value
        else:
            raise ValueError(f"Share_matrix input error, type of input: {type(matrix_tensor)}")
        dest_role = consts.GUEST if self.role == consts.HOST else consts.HOST

        self.transfer_variable.share_matrix.remote(to_share, role=dest_role, suffix=curt_suffix)
        return random_tensor

    def received_share_matrix(self, cipher, q_field, encoder, suffix=tuple()):
        curt_suffix = ("share_matrix",) + suffix
        # share = self.transfer_variable.share_matrix.get_parties(parties=self.other_party,
        #                                                         suffix=curt_suffix)[0]
        dest_role = consts.GUEST if self.role == consts.HOST else consts.HOST

        share = self.transfer_variable.share_matrix.get(role=dest_role, idx=0,
                                                        suffix=curt_suffix)
        # return share.value

        if isinstance(share, np.ndarray):
            share = cipher.recursive_decrypt(share)
            share = encoder.encode(share)
            return fixedpoint_numpy.FixedPointTensor(value=share,
                                                     q_field=q_field,
                                                     endec=encoder)
        share = cipher.distribute_decrypt(share)
        share = encoder.encode(share)
        return fixedpoint_table.FixedPointTensor.from_value(share, q_field=q_field, encoder=encoder)

    def secure_matrix_mul_active(self, matrix, cipher, suffix=tuple()):
        curt_suffix = ("secure_matrix_mul",) + suffix
        dest_role = consts.GUEST if self.role == consts.HOST else consts.HOST
        if isinstance(matrix, fixedpoint_table.FixedPointTensor):
            de_matrix = self.fixpoint_encoder.decode(matrix.value)
            encrypt_mat = cipher.distribute_encrypt(de_matrix)
        else:
            encrypt_mat = cipher.recursive_encrypt(matrix.value)
        self.transfer_variable.share_matrix.remote(encrypt_mat, role=dest_role, idx=0, suffix=curt_suffix)
        # return self.received_share_matrix(cipher, q_field=self.fixpoint_encoder.n,
        #                                   encoder=self.fixpoint_encoder, suffix=suffix)

    def secure_matrix_mul_passive(self, matrix, suffix=tuple()):
        curt_suffix = ("secure_matrix_mul",) + suffix
        # share = self.transfer_variable.share_matrix.get_parties(parties=self.other_party,
        #                                                         suffix=curt_suffix)[0]
        dest_role = consts.GUEST if self.role == consts.HOST else consts.HOST

        share = self.transfer_variable.share_matrix.get(role=dest_role, idx=0,
                                                        suffix=curt_suffix)
        LOGGER.debug(f"suffix: {suffix}, share: {share}")
        if isinstance(share, np.ndarray) and len(share) == 0:
            xy = matrix.value.mapValues(lambda x: np.array([0]))
            xy = fixedpoint_table.FixedPointTensor.from_value(xy, q_field=matrix.q_field, encoder=matrix.endec)
        elif isinstance(share, np.ndarray):
            xy = matrix.dot_array(share, fit_intercept=self.fit_intercept)
        else:
            share_tensor = fixedpoint_table.PaillierFixedPointTensor.from_value(
                share, q_field=matrix.q_field, encoder=matrix.endec)
            xy = matrix.dot_local(share_tensor)
        LOGGER.debug(f"Finish dot")
        return self.share_matrix(xy, suffix=suffix)

    def secure_matrix_mul(self, matrix, cipher=None, suffix=tuple()):
        curt_suffix = ("secure_matrix_mul",) + suffix
        if cipher is not None:
            dest_role = consts.GUEST if self.role == consts.HOST else consts.HOST
            # LOGGER.debug(f"matrix.value: {matrix.value.first()}")
            # de_matrix = self.fixpoint_encoder.decode(matrix.value)
            # encrypt_mat = cipher.distribute_encrypt(de_matrix)
            if isinstance(matrix, fixedpoint_table.FixedPointTensor):
                de_matrix = self.fixpoint_encoder.decode(matrix.value)
                encrypt_mat = cipher.distribute_encrypt(de_matrix)
            else:
                encrypt_mat = cipher.recursive_encrypt(matrix.value)
            self.transfer_variable.share_matrix.remote(encrypt_mat, role=dest_role, idx=0, suffix=curt_suffix)
            return self.received_share_matrix(cipher, q_field=self.fixpoint_encoder.n,
                                              encoder=self.fixpoint_encoder, suffix=suffix)
        else:
            # share = self.transfer_variable.share_matrix.get_parties(parties=self.other_party,
            #                                                         suffix=curt_suffix)[0]
            dest_role = consts.GUEST if self.role == consts.HOST else consts.HOST
            share = self.transfer_variable.share_matrix.get(role=dest_role, idx=0,
                                                            suffix=curt_suffix)
            LOGGER.debug(f"Make share tensor")
            if isinstance(share, np.ndarray):
                xy = matrix.dot_array(share, fit_intercept=self.fit_intercept)
            else:
                share_tensor = fixedpoint_table.PaillierFixedPointTensor.from_value(
                    share, q_field=matrix.q_field, encoder=matrix.endec)
                xy = matrix.dot_local(share_tensor)
            LOGGER.debug(f"Finish dot")

            # xy = share_tensor
            return self.share_matrix(xy, suffix=suffix)
