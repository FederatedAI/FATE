#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import copy
from abc import ABC

import numpy as np

from fate_arch.session import get_parties
from federatedml.framework.hetero.procedure import batch_generator
from federatedml.linear_model.linear_model_base import BaseLinearModel
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.fixedpoint import FixedPointEndec
from federatedml.secureprotol.spdz.secure_matrix.secure_matrix import SecureMatrix
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy
from federatedml.transfer_variable.transfer_class.batch_generator_transfer_variable import \
    BatchGeneratorTransferVariable
from federatedml.transfer_variable.transfer_class.converge_checker_transfer_variable import \
    ConvergeCheckerTransferVariable
from federatedml.transfer_variable.transfer_class.sshe_model_transfer_variable import SSHEModelTransferVariable
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroSSHEBase(BaseLinearModel, ABC):
    def __init__(self):
        super().__init__()
        self.mode = consts.HETERO
        self.cipher = None
        self.q_field = None
        self.model_param = None
        self.labels = None
        self.batch_num = []
        self.secure_matrix_obj: SecureMatrix
        # self._set_parties()
        self.cipher_tool = None
        self.local_party = None
        self.other_party = None

    def _transfer_q_field(self):
        """
        if self.role == consts.GUEST:
            q_field = self.cipher.public_key.n
            self.transfer_variable.q_field.remote(q_field, role=consts.HOST, suffix=("q_field",))

        else:
            q_field = self.transfer_variable.q_field.get(role=consts.GUEST, idx=0,
                                                          suffix=("q_field",))

        return q_field
        """
        raise NotImplementedError(f"Should not be called here")

    def _init_model(self, params):
        super()._init_model(params)
        self.encrypted_mode_calculator_param = params.encrypted_mode_calculator_param
        if self.role == consts.HOST:
            self.init_param_obj.fit_intercept = False
        self.cipher = PaillierEncrypt()
        self.cipher.generate_key(self.model_param.encrypt_param.key_length)
        self.transfer_variable = SSHEModelTransferVariable()

        self.converge_func_name = params.early_stop
        self.reveal_every_iter = params.reveal_every_iter

        self.q_field = self._transfer_q_field()

        LOGGER.debug(f"q_field: {self.q_field}")

        if not self.reveal_every_iter:
            self.self_optimizer = copy.deepcopy(self.optimizer)
            self.remote_optimizer = copy.deepcopy(self.optimizer)

        self.batch_generator = batch_generator.Guest() if self.role == consts.GUEST else batch_generator.Host()
        self.batch_generator.register_batch_generator(BatchGeneratorTransferVariable(), has_arbiter=False)
        self.fixedpoint_encoder = FixedPointEndec(n=self.q_field)
        self.converge_transfer_variable = ConvergeCheckerTransferVariable()
        self.secure_matrix_obj = SecureMatrix(party=self.local_party,
                                              q_field=self.q_field,
                                              other_party=self.other_party)

    def _init_weights(self, model_shape):
        return self.initializer.init_model(model_shape, init_params=self.init_param_obj)

    """
    def _set_parties(self):
        parties = []
        guest_parties = get_parties().roles_to_parties(["guest"])
        host_parties = get_parties().roles_to_parties(["host"])
        parties.extend(guest_parties)
        parties.extend(host_parties)

        local_party = get_parties().local_party
        other_party = parties[0] if parties[0] != local_party else parties[1]

        self.parties = parties
        self.local_party = local_party
        self.other_party = other_party
    """

    @property
    def is_respectively_reveal(self):
        return self.model_param.reveal_strategy == "respectively"

    def share_model(self, w, suffix):
        raise NotImplementedError("Should not be called here")

    def forward(self, weights, features, suffix, cipher):
        raise NotImplementedError("Should not be called here")

    def backward(self, error, features, suffix, cipher):
        raise NotImplementedError("Should not be called here")

    def compute_loss(self, weights, suffix, cipher):
        raise NotImplementedError("Should not be called here")

    def reveal_models(self, w_self, w_remote, suffix=None):
        raise NotImplementedError(f"Should not be called here")

    def check_converge_by_loss(self, loss, suffix):
        raise NotImplementedError(f"Should not be called here")

    def check_converge_by_weights(self, last_w, new_w, suffix):
        if self.reveal_every_iter:
            return self._reveal_every_iter_weights_check(last_w, new_w, suffix)
        else:
            return self._not_reveal_every_iter_weights_check(last_w, new_w, suffix)

    def _reveal_every_iter_weights_check(self, last_w, new_w, suffix):
        raise NotImplementedError("Should not be called here")

    def _not_reveal_every_iter_weights_check(self, last_w, new_w, suffix):
        last_w_self, last_w_remote = last_w
        w_self, w_remote = new_w
        grad_self = w_self - last_w_self
        grad_remote = w_remote - last_w_remote

        if self.role == consts.GUEST:
            grad_encode = np.hstack((grad_remote.value, grad_self.value))
        else:
            grad_encode = np.hstack((grad_self.value, grad_remote.value))

        grad_encode = np.array([grad_encode])

        grad_tensor_name = ".".join(("check_converge_grad",) + suffix)
        grad_tensor = fixedpoint_numpy.FixedPointTensor(value=grad_encode,
                                                        q_field=self.fixedpoint_encoder.n,
                                                        endec=self.fixedpoint_encoder,
                                                        tensor_name=grad_tensor_name)

        grad_tensor_transpose_name = ".".join(("check_converge_grad_transpose",) + suffix)
        grad_tensor_transpose = fixedpoint_numpy.FixedPointTensor(value=grad_encode.T,
                                                                  q_field=self.fixedpoint_encoder.n,
                                                                  endec=self.fixedpoint_encoder,
                                                                  tensor_name=grad_tensor_transpose_name)

        grad_norm_tensor_name = ".".join(("check_converge_grad_norm",) + suffix)

        grad_norm = grad_tensor.dot(grad_tensor_transpose, target_name=grad_norm_tensor_name).get()

        weight_diff = np.sqrt(grad_norm[0][0])
        LOGGER.info("iter: {}, weight_diff:{}, is_converged: {}".format(self.n_iter_,
                                                                        weight_diff, self.is_converged))
        is_converge = False
        if weight_diff < self.model_param.tol:
            is_converge = True
        return is_converge

    def get_single_model_param(self, model_weights=None, header=None):
        header = header if header else self.header
        result = {'iters': self.n_iter_,
                  'loss_history': self.loss_history,
                  'is_converged': self.is_converged,
                  'intercept': self.model_weights.intercept_,
                  'header': header,
                  'best_iteration': -1 if self.validation_strategy is None else
                  self.validation_strategy.best_iteration
                  }

        if self.role == consts.GUEST or self.is_respectively_reveal:
            model_weights = model_weights if model_weights else self.model_weights
            weight_dict = {}
            for idx, header_name in enumerate(header):
                coef_i = model_weights.coef_[idx]
                weight_dict[header_name] = coef_i

            result['weight'] = weight_dict

        return result

    def load_single_model(self, single_model_obj):
        LOGGER.info("start to load single model")

        if self.role == consts.GUEST or self.is_respectively_reveal:
            feature_shape = len(self.header)
            tmp_vars = np.zeros(feature_shape)
            weight_dict = dict(single_model_obj.weight)

            for idx, header_name in enumerate(self.header):
                tmp_vars[idx] = weight_dict.get(header_name)

            if self.fit_intercept:
                tmp_vars = np.append(tmp_vars, single_model_obj.intercept)
            self.model_weights = LinearModelWeights(tmp_vars, fit_intercept=self.fit_intercept)

        self.n_iter_ = single_model_obj.iters
        return self


class HeteroSSHEGuestBase(HeteroSSHEBase, ABC):
    def __init__(self):
        super().__init__()
        self.role = consts.GUEST
        self.local_party = get_parties().local_party
        self.other_party = get_parties().roles_to_parties(["host"])[0]

    def _transfer_q_field(self):
        q_field = self.cipher.public_key.n
        self.transfer_variable.q_field.remote(q_field, role=consts.HOST, suffix=("q_field",))

        return q_field

    def share_model(self, w, suffix):
        source = [w, self.other_party]
        wb, wa = (
            fixedpoint_numpy.FixedPointTensor.from_source(f"wb_{suffix}", source[0],
                                                          encoder=self.fixedpoint_encoder,
                                                          q_field=self.q_field),
            fixedpoint_numpy.FixedPointTensor.from_source(f"wa_{suffix}", source[1],
                                                          encoder=self.fixedpoint_encoder,
                                                          q_field=self.q_field),
        )
        return wb, wa

    def reveal_models(self, w_self, w_remote, suffix=None):
        if suffix is None:
            suffix = self.n_iter_

        if self.model_param.reveal_strategy == "respectively":

            new_w = w_self.get(tensor_name=f"wb_{suffix}",
                               broadcast=False)
            w_remote.broadcast_reconstruct_share(tensor_name=f"wa_{suffix}")

        elif self.model_param.reveal_strategy == "encrypted_reveal_in_host":

            new_w = w_self.get(tensor_name=f"wb_{suffix}",
                               broadcast=False)
            encrypted_w_remote = self.cipher.recursive_encrypt(self.fixedpoint_encoder.decode(w_remote.value))
            encrypted_w_remote_tensor = fixedpoint_numpy.PaillierFixedPointTensor(value=encrypted_w_remote)
            encrypted_w_remote_tensor.broadcast_reconstruct_share(tensor_name=f"wa_{suffix}")

        else:
            raise NotImplementedError(f"reveal strategy: {self.model_param.reveal_strategy} has not been implemented.")
        return new_w

    def check_converge_by_loss(self, loss, suffix):
        self.is_converged = self.converge_func.is_converge(loss)
        self.transfer_variable.is_converged.remote(self.is_converged, suffix=suffix)

        return self.is_converged


class HeteroSSHEHostBase(HeteroSSHEBase, ABC):
    def __init__(self):
        super().__init__()
        self.role = consts.HOST
        self.local_party = get_parties().local_party
        self.other_party = get_parties().roles_to_parties(["guest"])[0]

    def _transfer_q_field(self):
        q_field = self.transfer_variable.q_field.get(role=consts.GUEST, idx=0,
                                                     suffix=("q_field",))

        return q_field

    def share_model(self, w, suffix):
        source = [w, self.other_party]
        wa, wb = (
            fixedpoint_numpy.FixedPointTensor.from_source(f"wa_{suffix}", source[0],
                                                          encoder=self.fixedpoint_encoder,
                                                          q_field=self.q_field),
            fixedpoint_numpy.FixedPointTensor.from_source(f"wb_{suffix}", source[1],
                                                          encoder=self.fixedpoint_encoder,
                                                          q_field=self.q_field),
        )
        return wa, wb

    def reveal_models(self, w_self, w_remote, suffix=None):
        if suffix is None:
            suffix = self.n_iter_

        if self.model_param.reveal_strategy == "respectively":
            w_remote.broadcast_reconstruct_share(tensor_name=f"wb_{suffix}")
            new_w = w_self.get(tensor_name=f"wa_{suffix}",
                               broadcast=False)

        elif self.model_param.reveal_strategy == "encrypted_reveal_in_host":
            w_remote.broadcast_reconstruct_share(tensor_name=f"wb_{suffix}")
            new_w = w_self.reconstruct(tensor_name=f"wa_{suffix}", broadcast=False)

        else:
            raise NotImplementedError(f"reveal strategy: {self.model_param.reveal_strategy} has not been implemented.")
        return new_w

    def check_converge_by_loss(self, loss, suffix):
        self.is_converged = self.transfer_variable.is_converged.get(idx=0, suffix=suffix)
        return self.is_converged
