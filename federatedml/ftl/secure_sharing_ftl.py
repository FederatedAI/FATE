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

from federatedml.ftl.plain_ftl import PlainFTLGuestModel, PlainFTLHostModel
from federatedml.ftl.secret_sharing_ops import share, local_compute_alpha_beta_share, compute_matmul_share, \
    compute_add_share, compute_sum_of_multiply_share


def create_mul_op_beaver_triple(As, Bs, Cs, *, is_party_a):
    print("As, Bs, Cs", As.shape, Bs.shape, Cs.shape)
    beaver_triple_share_map = dict()
    beaver_triple_share_map["is_party_a"] = is_party_a
    beaver_triple_share_map["As"] = As
    beaver_triple_share_map["Bs"] = Bs
    beaver_triple_share_map["Cs"] = Cs
    return beaver_triple_share_map


class EncryptedFTLGuestModel(PlainFTLGuestModel):

    def __init__(self, local_model, model_param, is_trace=False):
        super(EncryptedFTLGuestModel, self).__init__(local_model, model_param, is_trace)
        self.party_guest_bt_map = None
        self.mul_op_beaver_triple_share_map = None
        self.alpha_s = None
        self.beta_s = None

    def _reset_alpha_beta_share(self):
        self.alpha_s = None
        self.beta_s = None

    def _prepare_beaver_triple(self, global_index, op_id):
        # get beaver triple for operation:op_id at iteration:global_index
        A0 = self.party_guest_bt_map[global_index][op_id]["A0"]
        B0 = self.party_guest_bt_map[global_index][op_id]["B0"]
        C0 = self.party_guest_bt_map[global_index][op_id]["C0"]
        self.mul_op_beaver_triple_share_map = create_mul_op_beaver_triple(A0, B0, C0, is_party_a=True)

    def send_components(self):

        self._compute_components()
        y_overlap_phi_mapping_comp = self.y_overlap_phi + self.gamma * self.mapping_comp_A
        self.y_overlap_2_phi_2_g, y_overlap_2_phi_2_h = share(self.y_overlap_2_phi_2)
        self.y_overlap_phi_mapping_comp_g, y_overlap_phi_mapping_comp_h = share(y_overlap_phi_mapping_comp)

        # y_overlap_2_phi_2_h and omega_h to host
        return [y_overlap_2_phi_2_h, y_overlap_phi_mapping_comp_h]

    def receive_components(self, components):
        # receive uB_overlap_g, uB_overlap_2_g and mapping_comp_B_g from guest
        self.uB_overlap_g = components[0]
        self.uB_overlap_2_g = components[1]
        self.mapping_comp_B_g = components[2]

    def compute_shares_for_alpha_beta_for_overlap_uB_y_2_phi_2(self, global_index, op_id=None):
        op_id = "host_mul_op_1"
        self._prepare_beaver_triple(global_index, op_id)
        self.mul_op_beaver_triple_share_map["Xs"] = np.expand_dims(self.uB_overlap_g, axis=1)
        self.mul_op_beaver_triple_share_map["Ys"] = self.y_overlap_2_phi_2_g
        self.alpha_s, self.beta_s = local_compute_alpha_beta_share(self.mul_op_beaver_triple_share_map)
        return self.alpha_s, self.beta_s

    def compute_share_for_overlap_uB_y_2_phi_2(self, alpha_t, beta_t):
        overlap_uB_y_2_phi_2_g = compute_matmul_share(self.alpha_s, alpha_t, self.beta_s, beta_t,
                                                      self.mul_op_beaver_triple_share_map)
        self.overlap_uB_y_2_phi_2_g = np.squeeze(overlap_uB_y_2_phi_2_g, axis=1)
        self._reset_alpha_beta_share()
        return self.overlap_uB_y_2_phi_2_g

    def compute_share_for_overlap_federated_layer_grad(self):
        self.overlap_federated_layer_grad_g = compute_add_share(self.overlap_uB_y_2_phi_2_g,
                                                                self.y_overlap_phi_mapping_comp_g)
        return self.overlap_federated_layer_grad_g

    def compute_shares_for_alpha_beta_for_grad_W(self, overlap_grads_W_g, global_index, op_id=None):
        op_id = "host_mul_op_2"
        self._prepare_beaver_triple(global_index, op_id)

        overlap_federated_layer_grad_g = np.expand_dims(self.overlap_federated_layer_grad_g, axis=1)

        # TODO: probably do not need to do this. The numpy's broadcasting will do this implicitly
        overlap_federated_layer_grad_g = np.broadcast_to(overlap_federated_layer_grad_g, (
            overlap_grads_W_g.shape[0], overlap_grads_W_g.shape[1], overlap_grads_W_g.shape[2]))

        self.mul_op_beaver_triple_share_map["Xs"] = np.expand_dims(overlap_federated_layer_grad_g, axis=1)
        self.mul_op_beaver_triple_share_map["Ys"] = overlap_grads_W_g
        self.alpha_s, self.beta_s = local_compute_alpha_beta_share(self.mul_op_beaver_triple_share_map)
        return self.alpha_s, self.beta_s

    def compute_share_for_grad_W(self, alpha_t, beta_t):
        self.grad_W_g = compute_sum_of_multiply_share(self.alpha_s, alpha_t, self.beta_s, beta_t,
                                                      self.mul_op_beaver_triple_share_map, axis=0)
        self._reset_alpha_beta_share()
        return self.grad_W_g

    def compute_shares_for_alpha_beta_for_grad_b(self, overlap_grads_b_g, global_index, op_id=None):
        op_id = "host_mul_op_2"
        self._prepare_beaver_triple(global_index, op_id)
        self.mul_op_beaver_triple_share_map["Xs"] = np.expand_dims(self.overlap_federated_layer_grad_g, axis=1)
        self.mul_op_beaver_triple_share_map["Ys"] = overlap_grads_b_g
        self.alpha_s, self.beta_s = local_compute_alpha_beta_share(self.mul_op_beaver_triple_share_map)
        return self.alpha_s, self.beta_s

    def compute_share_for_grad_b(self, alpha_t, beta_t):
        self.grad_b_g = compute_sum_of_multiply_share(self.alpha_s, alpha_t, self.beta_s, beta_t,
                                                      self.mul_op_beaver_triple_share_map, axis=0)
        return self.grad_b_g

    # def _update_gradients(self):
    #
    #     # y_overlap_2 have shape (len(overlap_indexes), 1),
    #     # phi has shape (1, feature_dim),
    #     # y_overlap_2_phi has shape (len(overlap_indexes), 1, feature_dim)
    #     y_overlap_2_phi = np.expand_dims(self.y_overlap_2 * self.phi, axis=1)
    #
    #     # uB_2_overlap has shape (len(overlap_indexes), feature_dim, feature_dim)
    #     enc_y_overlap_2_phi_uB_overlap_2 = encrypt_matmul_3(y_overlap_2_phi, self.enc_uB_overlap_2)
    #     enc_loss_grads_const_part1 = np.sum(0.25 * np.squeeze(enc_y_overlap_2_phi_uB_overlap_2, axis=1), axis=0)
    #
    #     if self.is_trace:
    #         self.logger.debug("enc_y_overlap_2_phi_uB_overlap_2 shape" + str(enc_y_overlap_2_phi_uB_overlap_2.shape))
    #         self.logger.debug("enc_loss_grads_const_part1 shape" + str(enc_loss_grads_const_part1.shape))
    #
    #     y_overlap = np.tile(self.y_overlap, (1, self.enc_uB_overlap.shape[-1]))
    #     enc_loss_grads_const_part2 = compute_sum_XY(y_overlap * 0.5, self.enc_uB_overlap)
    #
    #     enc_const = enc_loss_grads_const_part1 - enc_loss_grads_const_part2
    #     enc_const_overlap = np.tile(enc_const, (len(self.overlap_indexes), 1))
    #     enc_const_nonoverlap = np.tile(enc_const, (len(self.non_overlap_indexes), 1))
    #     y_non_overlap = np.tile(self.y[self.non_overlap_indexes], (1, self.enc_uB_overlap.shape[-1]))
    #
    #     if self.is_trace:
    #         self.logger.debug("enc_const shape:" + str(enc_const.shape))
    #         self.logger.debug("enc_const_overlap shape" + str(enc_const_overlap.shape))
    #         self.logger.debug("enc_const_nonoverlap shape" + str(enc_const_nonoverlap.shape))
    #         self.logger.debug("y_non_overlap shape" + str(y_non_overlap.shape))
    #
    #     enc_grad_A_nonoverlap = compute_XY(self.alpha * y_non_overlap / len(self.y), enc_const_nonoverlap)
    #     enc_grad_A_overlap = compute_XY_plus_Z(self.alpha * y_overlap / len(self.y), enc_const_overlap,
    #                                            self.enc_mapping_comp_B)
    #
    #     if self.is_trace:
    #         self.logger.debug("enc_grad_A_nonoverlap shape" + str(enc_grad_A_nonoverlap.shape))
    #         self.logger.debug("enc_grad_A_overlap shape" + str(enc_grad_A_overlap.shape))
    #
    #     enc_loss_grad_A = [[0 for _ in range(self.enc_uB_overlap.shape[1])] for _ in range(len(self.y))]
    #     # TODO: need more efficient way to do following task
    #     for i, j in enumerate(self.non_overlap_indexes):
    #         enc_loss_grad_A[j] = enc_grad_A_nonoverlap[i]
    #     for i, j in enumerate(self.overlap_indexes):
    #         enc_loss_grad_A[j] = enc_grad_A_overlap[i]
    #
    #     enc_loss_grad_A = np.array(enc_loss_grad_A)
    #
    #     if self.is_trace:
    #         self.logger.debug("enc_loss_grad_A shape" + str(enc_loss_grad_A.shape))
    #         self.logger.debug("enc_loss_grad_A" + str(enc_loss_grad_A))
    #
    #     self.loss_grads = enc_loss_grad_A
    #     self.enc_grads_W, self.enc_grads_b = self.localModel.compute_encrypted_params_grads(
    #         self.X, enc_loss_grad_A)
    #
    # def send_gradients(self):
    #     return [self.enc_grads_W, self.enc_grads_b]
    #
    # def receive_gradients(self, gradients):
    #     self.localModel.apply_gradients(gradients)
    #
    # def send_loss(self):
    #     return self.loss
    #
    # def receive_loss(self, loss):
    #     self.loss = loss
    #
    # def _update_loss(self):
    #     uA_overlap_prime = - self.uA_overlap / self.feature_dim
    #     enc_loss_overlap = np.sum(compute_sum_XY(uA_overlap_prime, self.enc_uB_overlap))
    #     enc_loss_y = self.__compute_encrypt_loss_y(self.enc_uB_overlap, self.enc_uB_overlap_2, self.y_overlap, self.phi)
    #     self.loss = self.alpha * enc_loss_y + enc_loss_overlap
    #
    # def __compute_encrypt_loss_y(self, enc_uB_overlap, enc_uB_overlap_2, y_overlap, phi):
    #     enc_uB_phi = encrypt_matmul_2_ob(enc_uB_overlap, phi.transpose())
    #     enc_uB_2 = np.sum(enc_uB_overlap_2, axis=0)
    #     enc_phi_uB_2_Phi = encrypt_matmul_2_ob(encrypt_matmul_2_ob(phi, enc_uB_2), phi.transpose())
    #     enc_loss_y = (-0.5 * compute_sum_XY(y_overlap, enc_uB_phi)[0] + 1.0 / 8 * np.sum(enc_phi_uB_2_Phi)) + len(
    #         y_overlap) * np.log(2)
    #     return enc_loss_y
    #
    # def get_loss_grads(self):
    #     return self.loss_grads



class SecureSharingFTLHostModel(PlainFTLHostModel):

    def __init__(self, local_model, model_param, is_trace=False):
        super(SecureSharingFTLHostModel, self).__init__(local_model, model_param, is_trace)
        self.party_host_bt_map = None
        self.mul_op_beaver_triple_share_map = None
        self.uB_overlap_h = None
        self.uB_overlap_2_h = None
        self.mapping_comp_B_h = None
        self.alpha_s = None
        self.beta_s = None

    def _reset_alpha_beta_share(self):
        self.alpha_s = None
        self.beta_s = None

    def _prepare_beaver_triple(self, global_index, op_id):
        # get beaver triple for operation:op_id at iteration:global_index
        A1 = self.party_host_bt_map[global_index][op_id]["A1"]
        B1 = self.party_host_bt_map[global_index][op_id]["B1"]
        C1 = self.party_host_bt_map[global_index][op_id]["C1"]
        self.mul_op_beaver_triple_share_map = create_mul_op_beaver_triple(A1, B1, C1, is_party_a=False)

    def send_components(self):
        self._compute_components()
        uB_overlap_g, self.uB_overlap_h = share(self.uB_overlap)
        uB_overlap_2_g, self.uB_overlap_2_h = share(self.uB_overlap_2)
        mapping_comp_B_g, self.mapping_comp_B_h = share(self.mapping_comp_B)

        # uB_overlap_g, uB_overlap_2_g and mapping_comp_B_g to guest
        return [uB_overlap_g, uB_overlap_2_g, mapping_comp_B_g]

    def receive_components(self, components):
        # receive y_overlap_2_phi_2_h, y_overlap_phi_mapping_comp_h from host
        self.y_overlap_2_phi_2_h = components[0]
        self.y_overlap_phi_mapping_comp_h = components[1]

    def compute_shares_for_alpha_beta_for_overlap_uB_y_2_phi_2(self, global_index, op_id=None):
        op_id = "host_mul_op_1"
        self._prepare_beaver_triple(global_index, op_id)
        self.mul_op_beaver_triple_share_map["Xs"] = np.expand_dims(self.uB_overlap_h, axis=1)
        self.mul_op_beaver_triple_share_map["Ys"] = self.y_overlap_2_phi_2_h
        self.alpha_s, self.beta_s = local_compute_alpha_beta_share(self.mul_op_beaver_triple_share_map)
        return self.alpha_s, self.beta_s

    def compute_share_for_overlap_uB_y_2_phi_2(self, alpha_t, beta_t):
        overlap_uB_y_2_phi_2_h = compute_matmul_share(self.alpha_s, alpha_t, self.beta_s, beta_t,
                                                      self.mul_op_beaver_triple_share_map)
        self.overlap_uB_y_2_phi_2_h = np.squeeze(overlap_uB_y_2_phi_2_h, axis=1)
        self._reset_alpha_beta_share()
        return self.overlap_uB_y_2_phi_2_h

    def compute_share_for_overlap_federated_layer_grad(self):
        self.overlap_federated_layer_grad_h = compute_add_share(self.overlap_uB_y_2_phi_2_h,
                                                                self.y_overlap_phi_mapping_comp_h)
        return self.overlap_federated_layer_grad_h

    def compute_shares_for_local_gradients(self):
        overlap_grads_W, overlap_grads_b = self.localModel.compute_gradients(self.X[self.overlap_indexes])
        self.overlap_grads_W_g, self.overlap_grads_W_h = share(overlap_grads_W)
        self.overlap_grads_b_g, self.overlap_grads_b_h = share(overlap_grads_b)
        return self.overlap_grads_W_g, self.overlap_grads_b_g

    def compute_shares_for_alpha_beta_for_grad_W(self, global_index, op_id=None):
        op_id = "host_mul_op_2"

        overlap_federated_layer_grad_h = np.expand_dims(self.overlap_federated_layer_grad_h, axis=1)

        # TODO: probably do not need to do this. The numpy's broadcasting will do this implicitly
        overlap_federated_layer_grad_h = np.broadcast_to(overlap_federated_layer_grad_h, (
        self.overlap_grads_W_h.shape[0], self.overlap_grads_W_h.shape[1], self.overlap_grads_W_h.shape[2]))

        self._prepare_beaver_triple(global_index, op_id)
        self.mul_op_beaver_triple_share_map["Xs"] = np.expand_dims(overlap_federated_layer_grad_h, axis=1)
        self.mul_op_beaver_triple_share_map["Ys"] = self.overlap_grads_W_h
        self.alpha_s, self.beta_s = local_compute_alpha_beta_share(self.mul_op_beaver_triple_share_map)
        return self.alpha_s, self.beta_s

    def compute_share_for_grad_W(self, alpha_t, beta_t):
        self.grad_W_h = compute_sum_of_multiply_share(self.alpha_s, alpha_t, self.beta_s, beta_t,
                                                      self.mul_op_beaver_triple_share_map, axis=0)
        self._reset_alpha_beta_share()

    def compute_shares_for_alpha_beta_for_grad_b(self, global_index, op_id=None):
        op_id = "host_mul_op_2"
        self._prepare_beaver_triple(global_index, op_id)
        self.mul_op_beaver_triple_share_map["Xs"] = np.expand_dims(self.overlap_federated_layer_grad_h, axis=1)
        self.mul_op_beaver_triple_share_map["Ys"] = self.overlap_grads_b_h
        self.alpha_s, self.beta_s = local_compute_alpha_beta_share(self.mul_op_beaver_triple_share_map)
        return self.alpha_s, self.beta_s

    def compute_share_for_grad_b(self, alpha_t, beta_t):
        self.grad_b_h = compute_sum_of_multiply_share(self.alpha_s, alpha_t, self.beta_s, beta_t,
                                                      self.mul_op_beaver_triple_share_map, axis=0)

    def receive_gradients(self, gradients):
        # receive grad_W_g and grad_b_g and reconstruct grad_W and grad_b
        grad_W = gradients[0] + self.grad_W_h
        grad_b = gradients[1] + self.grad_b_h
        self.localModel.apply_gradients([grad_W, grad_b])

########################################################################################################################
    # def _update_gradients(self):
    #     uB_overlap_ex = np.expand_dims(self.uB_overlap, axis=1)
    #     enc_uB_overlap_y_overlap_2_phi_2 = encrypt_matmul_3(uB_overlap_ex, self.enc_y_overlap_2_phi_2)
    #     enc_l1_grad_B = compute_X_plus_Y(np.squeeze(enc_uB_overlap_y_overlap_2_phi_2, axis=1), self.enc_y_overlap_phi)
    #     enc_loss_grad_B = compute_X_plus_Y(self.alpha * enc_l1_grad_B, self.enc_mapping_comp_A)
    #
    #     self.loss_grads = enc_loss_grad_B
    #     self.enc_grads_W, self.enc_grads_b = self.localModel.compute_encrypted_params_grads(
    #         self.X[self.overlap_indexes], enc_loss_grad_B)
    #
    # def send_gradients(self):
    #     return [self.enc_grads_W, self.enc_grads_b]
    #
    # def receive_gradients(self, gradients):
    #     self.localModel.apply_gradients(gradients)
    #
    # def get_loss_grads(self):
    #     return self.loss_grads

# class LocalEncryptedFederatedTransferLearning(object):
#
#     def __init__(self, guest: EncryptedFTLGuestModel, host: EncryptedFTLHostModel, private_key=None):
#         super(LocalEncryptedFederatedTransferLearning, self).__init__()
#         self.guest = guest
#         self.host = host
#         self.private_key = private_key
#
#     def fit(self, X_A, X_B, y, overlap_indexes, non_overlap_indexes):
#         self.guest.set_batch(X_A, y, non_overlap_indexes, overlap_indexes)
#         self.host.set_batch(X_B, overlap_indexes)
#
#         comp_B = self.host.send_components()
#         comp_A = self.guest.send_components()
#
#         self.guest.receive_components(comp_B)
#         self.host.receive_components(comp_A)
#
#         encrypt_gradients_A = self.guest.send_gradients()
#         encrypt_gradients_B = self.host.send_gradients()
#
#         self.guest.receive_gradients(self.__decrypt_gradients(encrypt_gradients_A))
#         self.host.receive_gradients(self.__decrypt_gradients(encrypt_gradients_B))
#
#         encrypt_loss = self.guest.send_loss()
#         loss = self.__decrypt_loss(encrypt_loss)
#
#         return loss
#
#     def predict(self, X_B):
#         msg = self.host.predict(X_B)
#         return self.guest.predict(msg)
#
#     def __decrypt_gradients(self, encrypt_gradients):
#         return decrypt_matrix(self.private_key, encrypt_gradients[0]), decrypt_array(self.private_key,
#                                                                                      encrypt_gradients[1])
#
#     def __decrypt_loss(self, encrypt_loss):
#         return decrypt_scalar(self.private_key, encrypt_loss)
