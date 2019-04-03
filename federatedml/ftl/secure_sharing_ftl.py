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

from arch.api.utils import log_utils
from federatedml.ftl.secret_sharing_ops import share, local_compute_alpha_beta_share, compute_matmul_share, \
    compute_add_share, compute_sum_of_multiply_share

LOGGER = log_utils.getLogger()


def create_mul_op_beaver_triple_map(As, Bs, Cs, *, is_party_a):
    print("As, Bs, Cs", As.shape, Bs.shape, Cs.shape)
    beaver_triple_share_map = dict()
    beaver_triple_share_map["is_party_a"] = is_party_a
    beaver_triple_share_map["As"] = As
    beaver_triple_share_map["Bs"] = Bs
    beaver_triple_share_map["Cs"] = Cs
    return beaver_triple_share_map


class SecureSharingParty(object):
    def retrieve_beaver_triple(self, global_index, op_id) -> dict:
        pass


class MulAlphaBetaComputer(object):
    """
    tracking and computing shares for beaver triple, alpha and beta for a secure sharing party
    for each mul operation at each step.
    """
    def __init__(self, secure_sharing_party: SecureSharingParty):
        self.mul_op_beaver_triple_share_map = dict()
        self.mul_op_alpha_beta_map = dict()
        self.secure_sharing_party = secure_sharing_party

    def compute_shares_for_alpha_beta_for_mul_op(self, global_index, op_id, component_x, component_y):
        self.mul_op_beaver_triple_share_map[op_id] = self.secure_sharing_party.retrieve_beaver_triple(global_index,
                                                                                                      op_id)
        self.mul_op_beaver_triple_share_map[op_id]["Xs"] = component_x
        self.mul_op_beaver_triple_share_map[op_id]["Ys"] = component_y
        alpha_s, beta_s = local_compute_alpha_beta_share(self.mul_op_beaver_triple_share_map[op_id])
        self.mul_op_alpha_beta_map[op_id + "/alpha_s"] = alpha_s
        self.mul_op_alpha_beta_map[op_id + "/beta_s"] = beta_s
        return alpha_s, beta_s

    def get_shares_for_mul_op_alpha_beta(self, op_id):
        alpha_s = self.mul_op_alpha_beta_map[op_id + "/alpha_s"]
        beta_s = self.mul_op_alpha_beta_map[op_id + "/beta_s"]
        return alpha_s, beta_s

    def get_shares_for_mul_op_beaver_triple(self, op_id):
        return self.mul_op_beaver_triple_share_map[op_id]


class SecureSharingFTLGuestModel(SecureSharingParty):

    def __init__(self, local_model, model_param, is_trace=False):
        self.localModel = local_model
        self.feature_dim = local_model.get_encode_dim()
        self.alpha = model_param.alpha
        self.gamma = model_param.gamma
        self.is_trace = is_trace

        self.alpha_beta_computer = MulAlphaBetaComputer(self)
        self.party_guest_bt_map = None

        self.logger = LOGGER

    def set_batch(self, X, y, non_overlap_indexes=None, overlap_indexes=None):
        self.X = X
        self.y = y
        self.non_overlap_indexes = non_overlap_indexes
        self.overlap_indexes = overlap_indexes
        self.phi = None

    def set_bt_map(self, party_guest_bt_map):
        self.party_guest_bt_map = party_guest_bt_map

    @staticmethod
    def _compute_phi(uA, y):
        length_y = len(y)
        return np.expand_dims(np.sum(y * uA, axis=0) / length_y, axis=0)

    def _compute_components(self):
        self.uA = self.localModel.transform(self.X)
        # phi has shape (1, feature_dim)
        # phi_2 has shape (feature_dim, feature_dim)
        self.phi = self._compute_phi(self.uA, self.y)
        self.phi_2 = np.matmul(self.phi.transpose(), self.phi)

        # y_overlap and y_overlap_2 have shape (len(overlap_indexes), 1)
        self.y_overlap = self.y[self.overlap_indexes]
        self.y_overlap_2 = self.y_overlap * self.y_overlap

        if self.is_trace:
            self.logger.debug("phi shape" + str(self.phi.shape))
            self.logger.debug("phi_2 shape" + str(self.phi_2.shape))
            self.logger.debug("y_overlap shape" + str(self.y_overlap.shape))
            self.logger.debug("y_overlap_2 shape" + str(self.y_overlap_2.shape))

        # following two parameters will be sent to host
        # y_overlap_2_phi_2 has shape (len(overlap_indexes), feature_dim, feature_dim)
        # y_overlap_phi has shape (len(overlap_indexes), feature_dim)
        self.y_overlap_2_phi_2 = 0.25 * np.expand_dims(self.y_overlap_2, axis=2) * self.phi_2
        self.y_overlap_phi = -0.5 * self.y_overlap * self.phi

        self.uA_overlap = self.uA[self.overlap_indexes]
        # mapping_comp_A has shape (len(overlap_indexes), feature_dim)
        self.mapping_comp_A = - self.uA_overlap / self.feature_dim

        if self.is_trace:
            self.logger.debug("y_overlap_2_phi_2 shape" + str(self.y_overlap_2_phi_2.shape))
            self.logger.debug("y_overlap_phi shape" + str(self.y_overlap_phi.shape))
            self.logger.debug("mapping_comp_A shape" + str(self.mapping_comp_A.shape))

        # y_overlap_2_phi has shape (len(overlap_indexes), feature_dim)
        self.y_overlap_2_phi = 0.25 * self.y_overlap_2 * self.phi
        self.half_y_overlap = 0.5 * self.y_overlap

    def send_components(self):
        self._compute_components()
        y_overlap_phi_mapping_comp = self.y_overlap_phi + self.gamma * self.mapping_comp_A
        self.y_overlap_2_phi_2_g, y_overlap_2_phi_2_h = share(self.y_overlap_2_phi_2)
        self.y_overlap_phi_mapping_comp_g, y_overlap_phi_mapping_comp_h = share(y_overlap_phi_mapping_comp)
        self.y_overlap_2_phi_g, y_overlap_2_phi_h = share(self.y_overlap_2_phi)
        self.half_y_overlap_g, half_y_overlap_h = share(self.half_y_overlap)
        return [y_overlap_2_phi_2_h, y_overlap_phi_mapping_comp_h, y_overlap_2_phi_h, half_y_overlap_h]

    def receive_components(self, components):
        # receive uB_overlap_g, uB_overlap_2_g and mapping_comp_B_g from guest
        self.uB_overlap_g = components[0]
        self.uB_overlap_2_g = components[1]
        self.mapping_comp_B_g = components[2]

    def retrieve_beaver_triple(self, global_index, op_id):
        # get beaver triple for operation:op_id at iteration:global_index
        A0 = self.party_guest_bt_map[global_index][op_id]["A0"]
        B0 = self.party_guest_bt_map[global_index][op_id]["B0"]
        C0 = self.party_guest_bt_map[global_index][op_id]["C0"]
        # self.mul_op_beaver_triple_share_map[op_id] = create_mul_op_beaver_triple_map(A0, B0, C0, is_party_a=True)
        return create_mul_op_beaver_triple_map(A0, B0, C0, is_party_a=True)

    #
    #
    # computing components and gradients for guest
    #
    #
    # def compute_shares_for_alpha_beta_for_overlap_y_2_phi_uB_2(self, global_index, op_id=None):
    #     op_id = "mul_op_3"
    #     component_x = np.expand_dims(self.y_overlap_2_phi_g, axis=1)
    #     component_y = self.uB_overlap_2_g
    #     return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
    #                                                                              component_x, component_y)
    #
    # def compute_share_for_overlap_overlap_y_2_phi_uB_2(self, alpha_t, beta_t):
    #     op_id = "mul_op_3"
    #     alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
    #     overlap_y_2_phi_uB_2 = compute_matmul_share(alpha_s, alpha_t, beta_s, beta_t,
    #                                                   self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
    #                                                       op_id))
    #     # overlap_y_2_phi_uB_2 has shape(len(overlap_indexes), feature_dim)
    #     self.overlap_y_2_phi_uB_2 = np.squeeze(overlap_y_2_phi_uB_2, axis=1)
    #     return self.overlap_y_2_phi_uB_2
    #
    # def compute_shares_for_alpha_beta_for_overlap_y_uB(self, global_index, op_id=None):
    #     op_id = "mul_op_4"
    #     component_x = np.tile(self.half_y_overlap_g, (1, self.uB_overlap_g.shape[-1]))
    #     component_y = self.uB_overlap_g
    #     return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
    #                                                                              component_x, component_y)
    #
    # def compute_share_for_overlap_overlap_y_uB(self, alpha_t, beta_t):
    #     op_id = "mul_op_4"
    #     alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
    #     self.overlap_sum_y_uB = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
    #                                                           self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
    #                                                               op_id))
    #     return self.overlap_sum_y_uB

    #
    #
    # computing components and gradients for host
    #
    #
    def compute_shares_for_alpha_beta_for_overlap_uB_y_2_phi_2(self, global_index, op_id=None):
        op_id = "mul_op_0"
        component_x = np.expand_dims(self.uB_overlap_g, axis=1)
        component_y = self.y_overlap_2_phi_2_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_overlap_uB_y_2_phi_2(self, alpha_t, beta_t):
        op_id = "mul_op_0"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        overlap_uB_y_2_phi_2_g = compute_matmul_share(alpha_s, alpha_t, beta_s, beta_t,
                                                      self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                          op_id))
        self.overlap_uB_y_2_phi_2_g = np.squeeze(overlap_uB_y_2_phi_2_g, axis=1)
        return self.overlap_uB_y_2_phi_2_g

    def compute_share_for_overlap_federated_layer_grad(self):
        self.overlap_federated_layer_grad_g = compute_add_share(self.overlap_uB_y_2_phi_2_g,
                                                                self.y_overlap_phi_mapping_comp_g)
        return self.overlap_federated_layer_grad_g

    def compute_shares_for_alpha_beta_for_grad_W(self, overlap_grads_W_g, global_index, op_id=None):
        op_id = "mul_op_1"
        overlap_federated_layer_grad_g = np.expand_dims(self.overlap_federated_layer_grad_g, axis=1)

        # TODO: probably do not need to do this. The numpy's broadcasting will do this implicitly
        # overlap_federated_layer_grad_g = np.broadcast_to(overlap_federated_layer_grad_g, (
        #     overlap_grads_W_g.shape[0], overlap_grads_W_g.shape[1], overlap_grads_W_g.shape[2]))

        print("*** overlap_federated_layer_grad_g shape", overlap_federated_layer_grad_g.shape)
        component_x = overlap_federated_layer_grad_g
        component_y = overlap_grads_W_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_grad_W(self, alpha_t, beta_t):
        op_id = "mul_op_1"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.grad_W_g = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                      self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                          op_id), axis=0)
        return self.grad_W_g

    def compute_shares_for_alpha_beta_for_grad_b(self, overlap_grads_b_g, global_index, op_id=None):
        op_id = "mul_op_2"
        component_x = self.overlap_federated_layer_grad_g
        component_y = overlap_grads_b_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_grad_b(self, alpha_t, beta_t):
        op_id = "mul_op_2"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.grad_b_g = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                      self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                          op_id), axis=0)
        return self.grad_b_g


class SecureSharingFTLHostModel(SecureSharingParty):

    def __init__(self, local_model, model_param, is_trace=False):
        self.localModel = local_model
        self.feature_dim = local_model.get_encode_dim()
        self.alpha = model_param.alpha
        self.is_trace = is_trace

        self.party_host_bt_map = None
        # self.mul_op_beaver_triple_share_map = dict()
        # self.mul_op_alpha_beta_map = dict()
        self.alpha_beta_computer = MulAlphaBetaComputer(self)

        self.uB_overlap_h = None
        self.uB_overlap_2_h = None
        self.mapping_comp_B_h = None
        # self.alpha_s = None
        # self.beta_s = None
        self.logger = LOGGER

    def set_batch(self, X, overlap_indexes):
        self.X = X
        self.overlap_indexes = overlap_indexes

    def set_bt_map(self, party_host_bt_map):
        self.party_host_bt_map = party_host_bt_map

    def _compute_components(self):
        self.uB = self.localModel.transform(self.X)

        # following three parameters will be sent to guest
        # uB_overlap has shape (len(overlap_indexes), feature_dim)
        # uB_overlap_2 has shape (len(overlap_indexes), feature_dim, feature_dim)
        # mapping_comp_B has shape (len(overlap_indexes), feature_dim)
        self.uB_overlap = self.uB[self.overlap_indexes]
        self.uB_overlap_2 = np.matmul(np.expand_dims(self.uB_overlap, axis=2), np.expand_dims(self.uB_overlap, axis=1))
        self.mapping_comp_B = - self.uB_overlap / self.feature_dim

        if self.is_trace:
            self.logger.debug("uB_overlap shape" + str(self.uB_overlap.shape))
            self.logger.debug("uB_overlap_2 shape" + str(self.uB_overlap_2.shape))
            self.logger.debug("mapping_comp_B shape" + str(self.mapping_comp_B.shape))

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
        self.y_overlap_2_phi_h = components[2]
        self.half_y_overlap_h = components[3]

    def retrieve_beaver_triple(self, global_index, op_id):
        # get beaver triple for operation:op_id at iteration:global_index
        A1 = self.party_host_bt_map[global_index][op_id]["A1"]
        B1 = self.party_host_bt_map[global_index][op_id]["B1"]
        C1 = self.party_host_bt_map[global_index][op_id]["C1"]
        return create_mul_op_beaver_triple_map(A1, B1, C1, is_party_a=False)

    def compute_shares_for_alpha_beta_for_overlap_uB_y_2_phi_2(self, global_index, op_id=None):
        op_id = "mul_op_0"
        component_x = np.expand_dims(self.uB_overlap_h, axis=1)
        component_y = self.y_overlap_2_phi_2_h
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_overlap_uB_y_2_phi_2(self, alpha_t, beta_t):
        op_id = "mul_op_0"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        overlap_uB_y_2_phi_2_h = compute_matmul_share(alpha_s, alpha_t, beta_s, beta_t,
                                                      self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                          op_id))
        self.overlap_uB_y_2_phi_2_h = np.squeeze(overlap_uB_y_2_phi_2_h, axis=1)
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
        op_id = "mul_op_1"
        overlap_federated_layer_grad_h = np.expand_dims(self.overlap_federated_layer_grad_h, axis=1)
        # TODO: probably do not need to do this. The numpy's broadcasting will do this implicitly
        # overlap_federated_layer_grad_h = np.broadcast_to(overlap_federated_layer_grad_h, (
        # self.overlap_grads_W_h.shape[0], self.overlap_grads_W_h.shape[1], self.overlap_grads_W_h.shape[2]))
        print("*** overlap_federated_layer_grad_h shape", overlap_federated_layer_grad_h.shape)

        component_x = overlap_federated_layer_grad_h
        component_y = self.overlap_grads_W_h
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_grad_W(self, alpha_t, beta_t):
        op_id = "mul_op_1"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.grad_W_h = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                      self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                          op_id), axis=0)
        return self.grad_W_h

    def compute_shares_for_alpha_beta_for_grad_b(self, global_index, op_id=None):
        op_id = "mul_op_2"
        component_x = self.overlap_federated_layer_grad_h
        component_y = self.overlap_grads_b_h
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_grad_b(self, alpha_t, beta_t):
        op_id = "mul_op_2"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.grad_b_h = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                      self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                          op_id), axis=0)
        return self.grad_b_h

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


class LocalSecureSharingFederatedTransferLearning(object):

    def __init__(self, guest: SecureSharingFTLGuestModel, host: SecureSharingFTLHostModel, private_key=None):
        super(LocalSecureSharingFederatedTransferLearning, self).__init__()
        self.guest = guest
        self.host = host

    def create_beaver_triples(self):
        pass

    def fit(self, X_A, X_B, y, overlap_indexes, non_overlap_indexes, global_index):

        self.guest.set_batch(X_A, y, non_overlap_indexes, overlap_indexes)
        self.host.set_batch(X_B, overlap_indexes)

        comp_B = self.host.send_components()
        comp_A = self.guest.send_components()

        self.host.receive_components(comp_A)
        self.guest.receive_components(comp_B)

        alpha_h, beta_h = self.host.compute_shares_for_alpha_beta_for_overlap_uB_y_2_phi_2(global_index)
        alpha_g, beta_g = self.guest.compute_shares_for_alpha_beta_for_overlap_uB_y_2_phi_2(global_index)

        overlap_uB_y_2_phi_2_h = self.host.compute_share_for_overlap_uB_y_2_phi_2(alpha_g, beta_g)
        overlap_uB_y_2_phi_2_g = self.guest.compute_share_for_overlap_uB_y_2_phi_2(alpha_h, beta_h)

        overlap_federated_layer_grad_h = self.host.compute_share_for_overlap_federated_layer_grad()
        overlap_federated_layer_grad_g = self.guest.compute_share_for_overlap_federated_layer_grad()

        overlap_grads_W_g, overlap_grads_b_g = self.host.compute_shares_for_local_gradients()
        alpha_grad_W_h, beta_grad_W_h = self.host.compute_shares_for_alpha_beta_for_grad_W(global_index)
        alpha_grad_b_h, beta_grad_b_h = self.host.compute_shares_for_alpha_beta_for_grad_b(global_index)

        alpha_grad_W_g, beta_grad_W_g = self.guest.compute_shares_for_alpha_beta_for_grad_W(overlap_grads_W_g,
                                                                                            global_index)
        alpha_grad_b_g, beta_grad_b_g = self.guest.compute_shares_for_alpha_beta_for_grad_b(overlap_grads_b_g,
                                                                                            global_index)

        grad_W_h = self.host.compute_share_for_grad_W(alpha_grad_W_g, beta_grad_W_g)
        grad_W_g = self.guest.compute_share_for_grad_W(alpha_grad_W_h, beta_grad_W_h)

        grad_b_h = self.host.compute_share_for_grad_b(alpha_grad_b_g, beta_grad_b_g)
        grad_b_g = self.guest.compute_share_for_grad_b(alpha_grad_b_h, beta_grad_b_h)

        self.host.receive_gradients([grad_W_g, grad_b_g])



        # encrypt_gradients_A = self.guest.send_gradients()
        # encrypt_gradients_B = self.host.send_gradients()
        #
        # self.guest.receive_gradients(self.__decrypt_gradients(encrypt_gradients_A))
        # self.host.receive_gradients(self.__decrypt_gradients(encrypt_gradients_B))
        #
        # encrypt_loss = self.guest.send_loss()
        # loss = self.__decrypt_loss(encrypt_loss)
        #
        # return loss
    #
    # def predict(self, X_B):
    #     msg = self.host.predict(X_B)
    #     return self.guest.predict(msg)
    #
    # def __decrypt_gradients(self, encrypt_gradients):
    #     return decrypt_matrix(self.private_key, encrypt_gradients[0]), decrypt_array(self.private_key,
    #                                                                                  encrypt_gradients[1])
    #
    # def __decrypt_loss(self, encrypt_loss):
    #     return decrypt_scalar(self.private_key, encrypt_loss)
