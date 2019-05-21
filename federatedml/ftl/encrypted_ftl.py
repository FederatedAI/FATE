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

from federatedml.ftl.eggroll_computation.helper import distribute_compute_sum_XY, \
    distribute_compute_XY, distribute_encrypt_matrix, distribute_compute_XY_plus_Z, \
    distribute_encrypt_matmul_2_ob, distribute_encrypt_matmul_3, distribute_compute_X_plus_Y
from federatedml.ftl.encryption import encryption
from federatedml.ftl.encryption.encryption import decrypt_array, decrypt_matrix, decrypt_scalar
from federatedml.ftl.plain_ftl import PlainFTLGuestModel, PlainFTLHostModel


class EncryptedFTLGuestModel(PlainFTLGuestModel):

    def __init__(self, local_model, model_param, public_key=None, host_public_key=None, private_key=None,
                 is_min_gen_enc=True, is_trace=False):
        super(EncryptedFTLGuestModel, self).__init__(local_model, model_param, is_trace)
        self.public_key = public_key
        self.private_key = private_key
        self.host_public_key = host_public_key
        self.is_min_gen_enc = is_min_gen_enc

    def set_public_key(self, public_key):
        self.public_key = public_key

    def set_host_public_key(self, public_key):
        self.host_public_key = public_key

    def set_private_key(self, private_key):
        self.private_key = private_key

    def send_components(self):
        if self.is_min_gen_enc:
            self.logger.debug("using min_gen_enc")

            self._compute_components()

            # phi has shape (1, feature_dim)
            # phi_2 has shape (feature_dim, feature_dim)
            enc_phi = encryption.encrypt_matrix(self.public_key, self.phi)
            enc_phi_2 = distribute_encrypt_matmul_2_ob(self.phi.transpose(), enc_phi)

            # enc_y_overlap_2_phi_2 = 0.25 * np.expand_dims(self.y_overlap_2, axis=2) * enc_phi_2
            # enc_y_overlap_phi = -0.5 * self.y_overlap * enc_phi
            enc_y_overlap_2_phi_2 = distribute_compute_XY(0.25 * np.expand_dims(self.y_overlap_2, axis=2),
                                                          np.tile(enc_phi_2, (self.y_overlap_2.shape[0], 1, 1)))
            enc_y_overlap_phi = distribute_compute_XY(-0.5 * self.y_overlap, np.tile(enc_phi, (self.y_overlap.shape[0], 1)))
            enc_mapping_comp_A = distribute_encrypt_matrix(self.public_key, self.mapping_comp_A)

            return [enc_y_overlap_2_phi_2, enc_y_overlap_phi, enc_mapping_comp_A]
        else:
            components = super(EncryptedFTLGuestModel, self).send_components()
            return self.__encrypt_components(components)

    def __encrypt_components(self, components):
        enc_comp_0 = distribute_encrypt_matrix(self.public_key, components[0])
        enc_comp_1 = distribute_encrypt_matrix(self.public_key, components[1])
        enc_comp_2 = distribute_encrypt_matrix(self.public_key, components[2])
        return [enc_comp_0, enc_comp_1, enc_comp_2]

    def receive_components(self, components):
        self.enc_uB_overlap = components[0]
        self.enc_uB_overlap_2 = components[1]
        self.enc_mapping_comp_B = components[2]
        self._update_gradients()
        self._update_loss()

    def _update_gradients(self):

        # y_overlap_2 have shape (len(overlap_indexes), 1),
        # phi has shape (1, feature_dim),
        # y_overlap_2_phi has shape (len(overlap_indexes), 1, feature_dim)
        y_overlap_2_phi = np.expand_dims(self.y_overlap_2 * self.phi, axis=1)

        # uB_2_overlap has shape (len(overlap_indexes), feature_dim, feature_dim)
        enc_y_overlap_2_phi_uB_overlap_2 = distribute_encrypt_matmul_3(y_overlap_2_phi, self.enc_uB_overlap_2)
        enc_loss_grads_const_part1 = np.sum(0.25 * np.squeeze(enc_y_overlap_2_phi_uB_overlap_2, axis=1), axis=0)

        if self.is_trace:
            self.logger.debug("enc_y_overlap_2_phi_uB_overlap_2 shape" + str(enc_y_overlap_2_phi_uB_overlap_2.shape))
            self.logger.debug("enc_loss_grads_const_part1 shape" + str(enc_loss_grads_const_part1.shape))

        y_overlap = np.tile(self.y_overlap, (1, self.enc_uB_overlap.shape[-1]))
        enc_loss_grads_const_part2 = distribute_compute_sum_XY(y_overlap * 0.5, self.enc_uB_overlap)

        enc_const = enc_loss_grads_const_part1 - enc_loss_grads_const_part2
        enc_const_overlap = np.tile(enc_const, (len(self.overlap_indexes), 1))
        enc_const_nonoverlap = np.tile(enc_const, (len(self.non_overlap_indexes), 1))
        y_non_overlap = np.tile(self.y[self.non_overlap_indexes], (1, self.enc_uB_overlap.shape[-1]))

        if self.is_trace:
            self.logger.debug("enc_const shape:" + str(enc_const.shape))
            self.logger.debug("enc_const_overlap shape" + str(enc_const_overlap.shape))
            self.logger.debug("enc_const_nonoverlap shape" + str(enc_const_nonoverlap.shape))
            self.logger.debug("y_non_overlap shape" + str(y_non_overlap.shape))

        enc_grad_A_nonoverlap = distribute_compute_XY(self.alpha * y_non_overlap / len(self.y), enc_const_nonoverlap)
        enc_grad_A_overlap = distribute_compute_XY_plus_Z(self.alpha * y_overlap / len(self.y), enc_const_overlap,
                                                          self.enc_mapping_comp_B)

        if self.is_trace:
            self.logger.debug("enc_grad_A_nonoverlap shape" + str(enc_grad_A_nonoverlap.shape))
            self.logger.debug("enc_grad_A_overlap shape" + str(enc_grad_A_overlap.shape))

        enc_loss_grad_A = [[0 for _ in range(self.enc_uB_overlap.shape[1])] for _ in range(len(self.y))]
        # TODO: need more efficient way to do following task
        for i, j in enumerate(self.non_overlap_indexes):
            enc_loss_grad_A[j] = enc_grad_A_nonoverlap[i]
        for i, j in enumerate(self.overlap_indexes):
            enc_loss_grad_A[j] = enc_grad_A_overlap[i]

        enc_loss_grad_A = np.array(enc_loss_grad_A)

        if self.is_trace:
            self.logger.debug("enc_loss_grad_A shape" + str(enc_loss_grad_A.shape))
            self.logger.debug("enc_loss_grad_A" + str(enc_loss_grad_A))

        self.loss_grads = enc_loss_grad_A
        self.enc_grads_W, self.enc_grads_b = self.localModel.compute_encrypted_params_grads(
            self.X, enc_loss_grad_A)

    def send_gradients(self):
        return [self.enc_grads_W, self.enc_grads_b]

    def receive_gradients(self, gradients):
        self.localModel.apply_gradients(gradients)

    def send_loss(self):
        return self.loss

    def receive_loss(self, loss):
        self.loss = loss

    def _update_loss(self):
        uA_overlap_prime = - self.uA_overlap / self.feature_dim
        enc_loss_overlap = np.sum(distribute_compute_sum_XY(uA_overlap_prime, self.enc_uB_overlap))
        enc_loss_y = self.__compute_encrypt_loss_y(self.enc_uB_overlap, self.enc_uB_overlap_2, self.y_overlap, self.phi)
        self.loss = self.alpha * enc_loss_y + enc_loss_overlap

    def __compute_encrypt_loss_y(self, enc_uB_overlap, enc_uB_overlap_2, y_overlap, phi):
        enc_uB_phi = distribute_encrypt_matmul_2_ob(enc_uB_overlap, phi.transpose())
        enc_uB_2 = np.sum(enc_uB_overlap_2, axis=0)
        enc_phi_uB_2_Phi = distribute_encrypt_matmul_2_ob(distribute_encrypt_matmul_2_ob(phi, enc_uB_2), phi.transpose())
        enc_loss_y = (-0.5 * distribute_compute_sum_XY(y_overlap, enc_uB_phi)[0] + 1.0 / 8 * np.sum(enc_phi_uB_2_Phi)) + len(
            y_overlap) * np.log(2)
        return enc_loss_y

    def get_loss_grads(self):
        return self.loss_grads


class EncryptedFTLHostModel(PlainFTLHostModel):

    def __init__(self, local_model, model_param, public_key=None, guest_public_key=None, private_key=None,
                 is_min_gen_enc=True, is_trace=False):
        super(EncryptedFTLHostModel, self).__init__(local_model, model_param, is_trace)
        self.public_key = public_key
        self.private_key = private_key
        self.guest_public_key = guest_public_key
        self.is_min_gen_enc = is_min_gen_enc

    def set_public_key(self, public_key):
        self.public_key = public_key

    def set_guest_public_key(self, public_key):
        self.guest_public_key = public_key

    def set_private_key(self, private_key):
        self.private_key = private_key

    def send_components(self):
        if self.is_min_gen_enc:
            self.logger.debug("using min_gen_enc")

            self._compute_components()

            # enc_uB_overlap has shape (len(overlap_indexes), feature_dim)
            # enc_uB_overlap_2 has shape (len(overlap_indexes), feature_dim, feature_dim)
            enc_uB_overlap = distribute_encrypt_matrix(self.public_key, self.uB_overlap)
            enc_uB_overlap_2 = distribute_encrypt_matmul_3(np.expand_dims(self.uB_overlap, axis=2),
                                                           np.expand_dims(enc_uB_overlap, axis=1))

            # enc_mapping_comp_B has shape (len(overlap_indexes), feature_dim)
            scale_factor = np.tile((-1 / self.feature_dim), (enc_uB_overlap.shape[0], enc_uB_overlap.shape[1]))
            enc_mapping_comp_B = distribute_compute_XY(enc_uB_overlap, scale_factor)
            # enc_mapping_comp_B = enc_uB_overlap * (-1 / self.feature_dim)
            # enc_mapping_comp_B = encrypt_matrix(self.public_key, self.mapping_comp_B)

            return [enc_uB_overlap, enc_uB_overlap_2, enc_mapping_comp_B]
        else:
            components = super(EncryptedFTLHostModel, self).send_components()
            return self.__encrypt_components(components)

    def __encrypt_components(self, components):
        enc_comp_0 = distribute_encrypt_matrix(self.public_key, components[0])
        enc_comp_1 = distribute_encrypt_matrix(self.public_key, components[1])
        enc_comp_2 = distribute_encrypt_matrix(self.public_key, components[2])
        return [enc_comp_0, enc_comp_1, enc_comp_2]

    def receive_components(self, components):
        self.enc_y_overlap_2_phi_2 = components[0]
        self.enc_y_overlap_phi = components[1]
        self.enc_mapping_comp_A = components[2]
        self._update_gradients()

    def _update_gradients(self):
        uB_overlap_ex = np.expand_dims(self.uB_overlap, axis=1)
        enc_uB_overlap_y_overlap_2_phi_2 = distribute_encrypt_matmul_3(uB_overlap_ex, self.enc_y_overlap_2_phi_2)
        enc_l1_grad_B = distribute_compute_X_plus_Y(np.squeeze(enc_uB_overlap_y_overlap_2_phi_2, axis=1), self.enc_y_overlap_phi)
        enc_loss_grad_B = distribute_compute_X_plus_Y(self.alpha * enc_l1_grad_B, self.enc_mapping_comp_A)

        self.loss_grads = enc_loss_grad_B
        self.enc_grads_W, self.enc_grads_b = self.localModel.compute_encrypted_params_grads(
            self.X[self.overlap_indexes], enc_loss_grad_B)

    def send_gradients(self):
        return [self.enc_grads_W, self.enc_grads_b]

    def receive_gradients(self, gradients):
        self.localModel.apply_gradients(gradients)

    def get_loss_grads(self):
        return self.loss_grads


class LocalEncryptedFederatedTransferLearning(object):

    def __init__(self, guest: EncryptedFTLGuestModel, host: EncryptedFTLHostModel, private_key=None):
        super(LocalEncryptedFederatedTransferLearning, self).__init__()
        self.guest = guest
        self.host = host
        self.private_key = private_key

    def fit(self, X_A, X_B, y, overlap_indexes, non_overlap_indexes):
        self.guest.set_batch(X_A, y, non_overlap_indexes, overlap_indexes)
        self.host.set_batch(X_B, overlap_indexes)

        comp_B = self.host.send_components()
        comp_A = self.guest.send_components()

        self.guest.receive_components(comp_B)
        self.host.receive_components(comp_A)

        encrypt_gradients_A = self.guest.send_gradients()
        encrypt_gradients_B = self.host.send_gradients()

        self.guest.receive_gradients(self.__decrypt_gradients(encrypt_gradients_A))
        self.host.receive_gradients(self.__decrypt_gradients(encrypt_gradients_B))

        encrypt_loss = self.guest.send_loss()
        loss = self.__decrypt_loss(encrypt_loss)

        return loss

    def predict(self, X_B):
        msg = self.host.predict(X_B)
        return self.guest.predict(msg)

    def __decrypt_gradients(self, encrypt_gradients):
        return decrypt_matrix(self.private_key, encrypt_gradients[0]), decrypt_array(self.private_key,
                                                                                     encrypt_gradients[1])

    def __decrypt_loss(self, encrypt_loss):
        return decrypt_scalar(self.private_key, encrypt_loss)
