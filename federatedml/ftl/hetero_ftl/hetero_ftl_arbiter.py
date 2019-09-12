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

import time

from arch.api.utils import log_utils
from federatedml.ftl.eggroll_computation.helper import distribute_decrypt_matrix
from federatedml.ftl.encryption.encryption import decrypt_scalar, decrypt_array
from federatedml.ftl.hetero_ftl.hetero_ftl_base import HeteroFTLParty
from federatedml.optim.convergence import AbsConverge
from federatedml.param.ftl_param import FTLModelParam
from federatedml.secureprotol.encrypt import PaillierEncrypt
from federatedml.util import consts
from federatedml.transfer_variable.transfer_class.hetero_ftl_transfer_variable import HeteroFTLTransferVariable

LOGGER = log_utils.getLogger()


class HeteroFTLArbiter(HeteroFTLParty):
    def __init__(self, model_param: FTLModelParam):
        super(HeteroFTLArbiter, self).__init__()
        self.max_iter = model_param.max_iter
        self.converge_func = AbsConverge(model_param.eps)

        self.transfer_variable = HeteroFTLTransferVariable()
        self.n_iter_ = 0
        self.private_key = None

    def fit(self):
        LOGGER.info("@ start arbiter fit")
        paillierEncrypt = PaillierEncrypt()
        paillierEncrypt.generate_key()
        public_key = paillierEncrypt.get_public_key()
        private_key = paillierEncrypt.get_privacy_key()
        self.private_key = private_key

        # distribute public key to guest and host
        self._do_remote(public_key,
                        name=self.transfer_variable.paillier_pubkey.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey),
                        role=consts.HOST,
                        idx=-1)
        self._do_remote(public_key,
                        name=self.transfer_variable.paillier_pubkey.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey),
                        role=consts.GUEST,
                        idx=-1)

        is_stop = False
        start_time = time.time()
        while self.n_iter_ < self.max_iter:
            # LOGGER.debug("@ iteration: " + str(self.n_iter_))
            # decrypt gradient from host
            encrypt_host_gradient = self._do_get(name=self.transfer_variable.encrypt_host_gradient.name,
                                                 tag=self.transfer_variable.generate_transferid(
                                                     self.transfer_variable.encrypt_host_gradient, self.n_iter_),
                                                 idx=-1)[0]

            decrypt_host_gradient = self.__decrypt_gradients(encrypt_host_gradient)
            self._do_remote(decrypt_host_gradient,
                            name=self.transfer_variable.decrypt_host_gradient.name,
                            tag=self.transfer_variable.generate_transferid(
                                self.transfer_variable.decrypt_host_gradient, self.n_iter_),
                            role=consts.HOST,
                            idx=-1)

            # decrypt gradient from guest
            encrypt_guest_gradient = self._do_get(name=self.transfer_variable.encrypt_guest_gradient.name,
                                                  tag=self.transfer_variable.generate_transferid(
                                                      self.transfer_variable.encrypt_guest_gradient, self.n_iter_),
                                                  idx=-1)[0]

            decrypt_guest_gradient = self.__decrypt_gradients(encrypt_guest_gradient)
            self._do_remote(decrypt_guest_gradient,
                            name=self.transfer_variable.decrypt_guest_gradient.name,
                            tag=self.transfer_variable.generate_transferid(
                                self.transfer_variable.decrypt_guest_gradient, self.n_iter_),
                            role=consts.GUEST,
                            idx=-1)

            # decrypt loss from guest
            encrypt_loss = self._do_get(name=self.transfer_variable.encrypt_loss.name,
                                        tag=self.transfer_variable.generate_transferid(
                                            self.transfer_variable.encrypt_loss, self.n_iter_),
                                        idx=-1)[0]

            loss = self.__decrypt_loss(encrypt_loss)
            if self.converge_func.is_converge(loss):
                is_stop = True

            # send is_stop indicator to host and guest
            self._do_remote(is_stop,
                            name=self.transfer_variable.is_encrypted_ftl_stopped.name,
                            tag=self.transfer_variable.generate_transferid(
                                self.transfer_variable.is_encrypted_ftl_stopped, self.n_iter_),
                            role=consts.HOST,
                            idx=-1)
            self._do_remote(is_stop,
                            name=self.transfer_variable.is_encrypted_ftl_stopped.name,
                            tag=self.transfer_variable.generate_transferid(
                                self.transfer_variable.is_encrypted_ftl_stopped, self.n_iter_),
                            role=consts.GUEST,
                            idx=-1)

            LOGGER.info("@ time: " + str(time.time()) + ", ep: " + str(self.n_iter_) + ", loss: " + str(loss))
            LOGGER.info("@ converged: " + str(is_stop))
            self.n_iter_ += 1
            if is_stop:
                break

        end_time = time.time()
        LOGGER.info("@ running time: " + str(end_time - start_time))

    def __decrypt_gradients(self, encrypt_gradients):
        return distribute_decrypt_matrix(self.private_key, encrypt_gradients[0]), decrypt_array(self.private_key,
                                                                                                encrypt_gradients[1])

    def __decrypt_loss(self, encrypt_loss):
        return decrypt_scalar(self.private_key, encrypt_loss)
