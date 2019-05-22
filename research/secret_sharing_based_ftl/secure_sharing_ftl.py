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
from research.beaver_triples_generation.secret_sharing_ops import share, local_compute_alpha_beta_share, compute_matmul_share, \
    compute_add_share, compute_minus_share, compute_sum_of_multiply_share, compute_multiply_share

LOGGER = log_utils.getLogger()


def create_mul_op_beaver_triple_map(As, Bs, Cs, *, is_party_a):
    # print("As, Bs, Cs", As.shape, Bs.shape, Cs.shape)
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


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


#
# Guest model for secure sharing based federated transfer learning
#
class SecureSharingFTLGuestModel(SecureSharingParty):

    def __init__(self, local_model, model_param, is_trace=False):
        self.localModel = local_model
        self.feature_dim = local_model.get_encode_dim()
        self.alpha = model_param.alpha
        self.gamma = model_param.gamma
        self.l2_param = model_param.l2_param
        self.is_trace = is_trace

        self.alpha_beta_computer = MulAlphaBetaComputer(self)
        self.party_guest_bt_map = None
        self.logger = LOGGER

    def set_batch(self, X, y, *, overlap_indexes=None, guest_non_overlap_indexes=None):
        self.X = X
        self.y = y
        self.guest_non_overlap_indexes = guest_non_overlap_indexes
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
        model_parameters = self.localModel.get_model_parameters()
        self.Wh = model_parameters["Wh"]

        # phi has shape (1, feature_dim)
        # phi_2 has shape (feature_dim, feature_dim)
        self.phi = self._compute_phi(self.uA, self.y)
        self.phi_2 = np.matmul(self.phi.transpose(), self.phi)

        # y_overlap and y_overlap_2 have shape (len(overlap_indexes), 1)
        self.y_overlap = self.y[self.overlap_indexes]
        self.y_overlap_2 = self.y_overlap * self.y_overlap

        self.ave_y_overlap = self.y_overlap / len(self.y)
        self.ave_y_non_overlap = self.y[self.guest_non_overlap_indexes] / len(self.y)

        if self.is_trace:
            self.logger.debug("phi shape" + str(self.phi.shape))
            self.logger.debug("phi_2 shape" + str(self.phi_2.shape))
            self.logger.debug("y_overlap shape" + str(self.y_overlap.shape))
            self.logger.debug("y_overlap_2 shape" + str(self.y_overlap_2.shape))

        # following two parameters will be sent to host
        # y_overlap_2_phi_2 has shape (len(overlap_indexes), feature_dim, feature_dim)
        # y_overlap_phi has shape (len(overlap_indexes), feature_dim)
        self.y_overlap_2_phi_2 = 0.25 * np.expand_dims(self.y_overlap_2, axis=2) * self.phi_2
        self.y_overlap_phi = - 0.5 * self.y_overlap * self.phi

        self.uA_overlap = self.uA[self.overlap_indexes]
        # mapping_comp_A has shape (len(overlap_indexes), feature_dim)
        # self.mapping_comp_A = - self.uA_overlap / self.feature_dim
        self.mapping_comp_A = - self.uA_overlap

        if self.is_trace:
            self.logger.debug("y_overlap_2_phi_2 shape" + str(self.y_overlap_2_phi_2.shape))
            self.logger.debug("y_overlap_phi shape" + str(self.y_overlap_phi.shape))
            self.logger.debug("mapping_comp_A shape" + str(self.mapping_comp_A.shape))

        # y_overlap_2_phi has shape (len(overlap_indexes), feature_dim)
        self.y_overlap_2_phi = 0.25 * self.y_overlap_2 * self.phi
        self.half_y_overlap = 0.5 * self.y_overlap

    def send_components(self):
        self._compute_components()
        y_overlap_phi_mapping_comp = self.alpha * self.y_overlap_phi + self.gamma * self.mapping_comp_A
        self.y_overlap_2_phi_2_g, y_overlap_2_phi_2_h = share(self.y_overlap_2_phi_2)
        self.y_overlap_phi_mapping_comp_g, y_overlap_phi_mapping_comp_h = share(y_overlap_phi_mapping_comp)
        self.y_overlap_2_phi_g, y_overlap_2_phi_h = share(self.y_overlap_2_phi)
        self.half_y_overlap_g, half_y_overlap_h = share(self.half_y_overlap)
        self.y_overlap_phi_g, y_overlap_phi_h = share(self.y_overlap_phi)

        self.ave_y_overlap_g, ave_y_overlap_h = share(self.ave_y_overlap)
        self.ave_y_non_overlap_g, ave_y_non_overlap_h = share(self.ave_y_non_overlap)

        self.uA_overlap_g, uA_overlap_h = share(self.uA_overlap)

        self.phi_g, phi_h = share(self.phi)

        return [y_overlap_2_phi_2_h, y_overlap_phi_mapping_comp_h, y_overlap_2_phi_h, half_y_overlap_h, ave_y_overlap_h,
                ave_y_non_overlap_h, y_overlap_phi_h, uA_overlap_h, phi_h]

    def receive_components(self, components):
        # receive uB_overlap_g, uB_overlap_2_g and mapping_comp_B_g from guest
        self.uB_overlap_g = components[0]
        self.uB_overlap_2_g = components[1]
        self.mapping_comp_B_g = components[2]

    def retrieve_beaver_triple(self, global_index, op_id):
        # get beaver triple for operation:op_id at iteration:global_index

        A0 = self.party_guest_bt_map[0][op_id]["A0"]
        B0 = self.party_guest_bt_map[0][op_id]["B0"]
        C0 = self.party_guest_bt_map[0][op_id]["C0"]
        # A0 = self.party_guest_bt_map[global_index][op_id]["A0"]
        # B0 = self.party_guest_bt_map[global_index][op_id]["B0"]
        # C0 = self.party_guest_bt_map[global_index][op_id]["C0"]
        # self.mul_op_beaver_triple_share_map[op_id] = create_mul_op_beaver_triple_map(A0, B0, C0, is_party_a=True)
        return create_mul_op_beaver_triple_map(A0, B0, C0, is_party_a=True)

    #
    # guest side computes shares of components and gradients for federated layer of guest side
    #

    def compute_shares_for_alpha_beta_for_overlap_y_2_phi_uB_2(self, global_index, op_id=None):
        op_id = "mul_op_3"
        component_x = np.expand_dims(self.y_overlap_2_phi_g, axis=1)
        component_y = self.uB_overlap_2_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_overlap_y_2_phi_uB_2(self, alpha_t, beta_t):
        op_id = "mul_op_3"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        overlap_y_2_phi_uB_2 = compute_matmul_share(alpha_s, alpha_t, beta_s, beta_t,
                                                    self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                        op_id))
        # overlap_y_2_phi_uB_2 has shape(len(overlap_indexes), feature_dim)
        self.overlap_y_2_phi_uB_2_g = np.squeeze(overlap_y_2_phi_uB_2, axis=1)
        return self.overlap_y_2_phi_uB_2_g

    def compute_shares_for_alpha_beta_for_overlap_y_uB(self, global_index, op_id=None):
        op_id = "mul_op_4"
        component_x = np.tile(self.half_y_overlap_g, (1, self.uB_overlap_g.shape[-1]))
        component_y = self.uB_overlap_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_overlap_y_uB(self, alpha_t, beta_t):
        op_id = "mul_op_4"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.overlap_y_uB_g = compute_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                     self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                         op_id))
        return self.overlap_y_uB_g

    def compute_const_part(self):
        # compute share_for_overlap_y_2_phi_uB_2_minus_y_uB
        self.const_g = compute_minus_share(np.sum(self.overlap_y_2_phi_uB_2_g, axis=0),
                                           np.sum(self.overlap_y_uB_g, axis=0))
        return self.const_g

    def compute_shares_for_alpha_beta_for_guest_non_overlap_federated_layer_grad(self, global_index, op_id=None):
        op_id = "mul_op_5"
        component_x = self.const_g
        component_y = self.ave_y_non_overlap_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_guest_non_overlap_federated_layer_grad(self, alpha_t, beta_t):
        op_id = "mul_op_5"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        guest_non_overlap_federated_layer_grad_g = compute_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                                               self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                                                   op_id))
        self.guest_non_overlap_federated_layer_grad_g = self.alpha * guest_non_overlap_federated_layer_grad_g
        return self.guest_non_overlap_federated_layer_grad_g

    def compute_shares_for_alpha_beta_for_guest_part_overlap_federated_layer_grad(self, global_index, op_id=None):
        op_id = "mul_op_6"
        component_x = self.const_g
        component_y = self.ave_y_overlap_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_guest_part_overlap_federated_layer_grad(self, alpha_t, beta_t):
        op_id = "mul_op_6"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.guest_part_overlap_federated_layer_grad_g = compute_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                                                self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                                                    op_id))
        return self.guest_part_overlap_federated_layer_grad_g

    def compute_share_for_guest_overlap_federated_layer_grad(self):
        self.guest_overlap_federated_layer_grad_g = compute_add_share(self.alpha * self.guest_part_overlap_federated_layer_grad_g,
                                                                      self.gamma * self.mapping_comp_B_g)
        return self.guest_overlap_federated_layer_grad_g

    #
    # guest side computes shares of gradients for guest local model and updates guest local model
    #

    def compute_shares_for_guest_local_gradients(self):
        grads_W, grads_b = self.localModel.compute_gradients(self.X)
        self.guest_grads_W_g, self.guest_grads_W_h = share(grads_W)
        self.guest_grads_b_g, self.guest_grads_b_h = share(grads_b)
        return self.guest_grads_W_h, self.guest_grads_b_h

    def compute_shares_for_alpha_beta_for_guest_grad_W(self, global_index, op_id=None):
        op_id = "mul_op_1_for_guest"

        self.guest_federated_layer_grad_g = np.zeros((len(self.y), self.uB_overlap_g.shape[1]))
        self.guest_federated_layer_grad_g[self.guest_non_overlap_indexes, :] = self.guest_non_overlap_federated_layer_grad_g
        self.guest_federated_layer_grad_g[self.overlap_indexes, :] = self.guest_overlap_federated_layer_grad_g

        guest_federated_layer_grad_g = np.expand_dims(self.guest_federated_layer_grad_g, axis=1)
        # TODO: probably do not need to do this. The numpy's broadcasting will do this implicitly
        # overlap_federated_layer_grad_h = np.broadcast_to(overlap_federated_layer_grad_h, (
        # self.overlap_grads_W_h.shape[0], self.overlap_grads_W_h.shape[1], self.overlap_grads_W_h.shape[2]))
        # print("*** guest_federated_layer_grad_g shape", guest_federated_layer_grad_g.shape)

        component_x = guest_federated_layer_grad_g
        component_y = self.guest_grads_W_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_guest_grad_W(self, alpha_t, beta_t):
        op_id = "mul_op_1_for_guest"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.guest_loss_grads_W_g = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                                  self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                                      op_id), axis=0)
        return self.guest_loss_grads_W_g

    def compute_shares_for_alpha_beta_for_guest_grad_b(self, global_index, op_id=None):
        op_id = "mul_op_8"
        component_x = self.guest_federated_layer_grad_g
        component_y = self.guest_grads_b_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_guest_grad_b(self, alpha_t, beta_t):
        op_id = "mul_op_8"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.guest_loss_grads_b_g = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                                  self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                                      op_id), axis=0)
        return self.guest_loss_grads_b_g

    def receive_gradients(self, gradients):
        # receive grad_W_g and grad_b_g and reconstruct grad_W and grad_b
        grad_W = gradients[0] + self.guest_loss_grads_W_g
        grad_b = gradients[1] + self.guest_loss_grads_b_g
        # print("guest receive grad_W:", grad_W)
        # print("guest receive grad_b:", grad_b)
        grad_W = grad_W + self.l2_param * self.Wh
        self.localModel.apply_gradients([grad_W, grad_b])

    #
    # guest side computes share of loss
    #

    def compute_shares_for_alpha_beta_for_overlap_y_phi_uB(self, global_index, op_id=None):
        op_id = "mul_op_9"
        component_x = np.expand_dims(self.y_overlap_phi_g, axis=1)
        component_y = np.expand_dims(self.uB_overlap_g, axis=2)
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_overlap_y_phi_uB(self, alpha_t, beta_t):
        op_id = "mul_op_9"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.overlap_y_phi_uB_g = compute_matmul_share(alpha_s, alpha_t, beta_s, beta_t,
                                                       self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                           op_id))
        return self.overlap_y_phi_uB_g

    def compute_shares_for_alpha_beta_for_phi_overlap_uB_2(self, global_index, op_id=None):
        op_id = "mul_op_10"
        uB_overlap_2_g = np.sum(self.uB_overlap_2_g, axis=0)
        # phi has shape (1, hidden_dim)
        # uB_overlap_2_g has shape (hidden_dim, hidden_dim)
        component_x = self.phi_g
        component_y = uB_overlap_2_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_phi_overlap_uB_2(self, alpha_t, beta_t):
        op_id = "mul_op_10"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.phi_overlap_uB_2_g = compute_matmul_share(alpha_s, alpha_t, beta_s, beta_t,
                                                     self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                         op_id))
        return self.phi_overlap_uB_2_g

    def compute_shares_for_alpha_beta_for_phi_overlap_uB_2_phi(self, global_index, op_id=None):
        op_id = "mul_op_11"
        # phi_overlap_uB_2 has shape (1, hidden_dim)
        # phi.transpose() has shape (hidden_dim, 1)
        component_x = self.phi_overlap_uB_2_g
        component_y = self.phi_g.transpose()
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_phi_overlap_uB_2_phi(self, alpha_t, beta_t):
        op_id = "mul_op_11"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.phi_overlap_uB_2_phi_g = compute_matmul_share(alpha_s, alpha_t, beta_s, beta_t,
                                                           self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                               op_id))
        return self.phi_overlap_uB_2_phi_g

    def compute_shares_for_alpha_beta_for_uA_uB(self, global_index, op_id=None):
        op_id = "mul_op_12"
        uA_overlap_g = - self.uA_overlap_g / self.feature_dim
        # uA_overlap_g has shape (N_AB, hidden_dim)
        # uB_overlap_g has shape (N_AB, hidden_dim)
        component_x = uA_overlap_g
        component_y = self.uB_overlap_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_uA_uB(self, alpha_t, beta_t):
        op_id = "mul_op_12"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.uA_uB_g = compute_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                              self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                  op_id))
        return self.uA_uB_g

    def compute_share_for_loss(self):
        # print("-"*10)
        # print("self.overlap_y_phi_uB_g", np.sum(self.overlap_y_phi_uB_g))
        # print("self.phi_overlap_uB_2_phi_g", np.sum(self.phi_overlap_uB_2_phi_g))
        # print("self.uA_uB_g", np.sum(self.uA_uB_g))
        loss_g = (np.sum(self.overlap_y_phi_uB_g) + 1.0 / 8 * np.sum(self.phi_overlap_uB_2_phi_g)) + np.sum(self.uA_uB_g)
        # print("self.loss_g", loss_g)
        self.loss_g = loss_g + 0.5 * self.l2_param * np.sum(np.square(self.Wh))
        return self.loss_g, np.sum(self.overlap_y_phi_uB_g), np.sum(self.phi_overlap_uB_2_phi_g), np.sum(self.uA_uB_g)

    def compute_loss(self, loss_h):
        # output = sum(t ** 2) / 2
        self.loss = loss_h + self.loss_g
        return self.loss

    def predict(self, uB):
        if self.phi is None:
            self.uA = self.localModel.transform(self.X)
            self.phi = self._compute_phi(self.uA, self.y)
        return sigmoid(np.matmul(uB, self.phi.transpose()))

    #
    # guest side computes shares of components and gradients for host side
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

    def compute_share_for_host_overlap_federated_layer_grad(self):
        self.host_overlap_federated_layer_grad_g = compute_add_share(self.alpha * self.overlap_uB_y_2_phi_2_g,
                                                                     self.y_overlap_phi_mapping_comp_g)
        return self.host_overlap_federated_layer_grad_g

    def compute_shares_for_alpha_beta_for_host_grad_W(self, overlap_grads_W_g, global_index, op_id=None):
        op_id = "mul_op_1_for_host"
        overlap_federated_layer_grad_g = np.expand_dims(self.host_overlap_federated_layer_grad_g, axis=1)

        # TODO: probably do not need to do this. The numpy's broadcasting will do this implicitly
        # overlap_federated_layer_grad_g = np.broadcast_to(overlap_federated_layer_grad_g, (
        #     overlap_grads_W_g.shape[0], overlap_grads_W_g.shape[1], overlap_grads_W_g.shape[2]))

        # print("*** overlap_federated_layer_grad_g shape", overlap_federated_layer_grad_g, overlap_federated_layer_grad_g.shape)
        component_x = overlap_federated_layer_grad_g
        component_y = overlap_grads_W_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_host_grad_W(self, alpha_t, beta_t):
        op_id = "mul_op_1_for_host"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.host_grad_W_g = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                           self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                               op_id), axis=0)
        return self.host_grad_W_g

    def compute_shares_for_alpha_beta_for_host_grad_b(self, overlap_grads_b_g, global_index, op_id=None):
        op_id = "mul_op_2"
        component_x = self.host_overlap_federated_layer_grad_g
        component_y = overlap_grads_b_g
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_host_grad_b(self, alpha_t, beta_t):
        op_id = "mul_op_2"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.host_grad_b_g = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                           self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                               op_id), axis=0)
        return self.host_grad_b_g


#
# Host model for secure sharing based federated transfer learning
#
class SecureSharingFTLHostModel(SecureSharingParty):

    def __init__(self, local_model, model_param, is_trace=False):
        self.localModel = local_model
        self.feature_dim = local_model.get_encode_dim()
        self.alpha = model_param.alpha
        self.gamma = model_param.gamma
        self.l2_param = model_param.l2_param
        self.is_trace = is_trace

        self.party_host_bt_map = None
        self.alpha_beta_computer = MulAlphaBetaComputer(self)

        self.uB_overlap_h = None
        self.uB_overlap_2_h = None
        self.mapping_comp_B_h = None
        # self.alpha_s = None
        # self.beta_s = None
        self.logger = LOGGER

    def set_batch(self, X, *, overlap_indexes=None, guest_non_overlap_indexes=None):
        self.X = X
        self.overlap_indexes = overlap_indexes
        self.guest_non_overlap_indexes = guest_non_overlap_indexes

    def set_bt_map(self, party_host_bt_map):
        self.party_host_bt_map = party_host_bt_map

    def _compute_components(self):
        self.uB = self.localModel.transform(self.X)
        model_parameters = self.localModel.get_model_parameters()
        self.Wh = model_parameters["Wh"]

        # following three parameters will be sent to guest
        # uB_overlap has shape (len(overlap_indexes), feature_dim)
        # uB_overlap_2 has shape (len(overlap_indexes), feature_dim, feature_dim)
        # mapping_comp_B has shape (len(overlap_indexes), feature_dim)
        # print("uB shape", self.uB.shape)
        self.uB_overlap = self.uB[self.overlap_indexes]
        # print("uB_overlap shape", self.uB_overlap.shape)
        self.uB_overlap_2 = np.matmul(np.expand_dims(self.uB_overlap, axis=2), np.expand_dims(self.uB_overlap, axis=1))
        # self.mapping_comp_B = - self.uB_overlap / self.feature_dim
        self.mapping_comp_B = - self.uB_overlap

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
        # receive y_overlap_2_phi_2_h, y_overlap_phi_mapping_comp_h, y_overlap_2_phi_h,
        # half_y_overlap_h, ave_y_overlap_h, ave_y_non_overlap_h from host
        self.y_overlap_2_phi_2_h = components[0]
        self.y_overlap_phi_mapping_comp_h = components[1]
        self.y_overlap_2_phi_h = components[2]
        self.half_y_overlap_h = components[3]
        self.ave_y_overlap_h = components[4]
        self.ave_y_non_overlap_h = components[5]
        self.y_overlap_phi_h = components[6]
        self.uA_overlap_h = components[7]
        self.phi_h = components[8]

    def retrieve_beaver_triple(self, global_index, op_id):
        # get beaver triple for operation:op_id at iteration:global_index
        A1 = self.party_host_bt_map[0][op_id]["A1"]
        B1 = self.party_host_bt_map[0][op_id]["B1"]
        C1 = self.party_host_bt_map[0][op_id]["C1"]
        # A1 = self.party_host_bt_map[global_index][op_id]["A1"]
        # B1 = self.party_host_bt_map[global_index][op_id]["B1"]
        # C1 = self.party_host_bt_map[global_index][op_id]["C1"]
        return create_mul_op_beaver_triple_map(A1, B1, C1, is_party_a=False)

    #
    # host side computes shares of components and gradients for federated layer of host side
    #

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

    def compute_share_for_host_overlap_federated_layer_grad(self):
        self.host_overlap_federated_layer_grad_h = compute_add_share(self.alpha * self.overlap_uB_y_2_phi_2_h,
                                                                     self.y_overlap_phi_mapping_comp_h)
        return self.host_overlap_federated_layer_grad_h

    #
    # host side computes shares of gradients for host local model and updates host local model
    #

    def compute_shares_for_host_local_gradients(self):
        # print("**# overlap_indexes", len(self.overlap_indexes))
        overlap_grads_W, overlap_grads_b = self.localModel.compute_gradients(self.X[self.overlap_indexes])
        # print("**# overlap_grads_W shape", overlap_grads_W.shape)
        self.host_overlap_grads_W_g, self.host_overlap_grads_W_h = share(overlap_grads_W)
        self.host_overlap_grads_b_g, self.host_overlap_grads_b_h = share(overlap_grads_b)
        return self.host_overlap_grads_W_g, self.host_overlap_grads_b_g

    def compute_shares_for_alpha_beta_for_host_grad_W(self, global_index, op_id=None):
        op_id = "mul_op_1_for_host"
        overlap_federated_layer_grad_h = np.expand_dims(self.host_overlap_federated_layer_grad_h, axis=1)
        # TODO: probably do not need to do this. The numpy's broadcasting will do this implicitly
        # overlap_federated_layer_grad_h = np.broadcast_to(overlap_federated_layer_grad_h, (
        # self.overlap_grads_W_h.shape[0], self.overlap_grads_W_h.shape[1], self.overlap_grads_W_h.shape[2]))
        # print("*** overlap_federated_layer_grad_h shape", overlap_federated_layer_grad_h, overlap_federated_layer_grad_h.shape)

        component_x = overlap_federated_layer_grad_h
        component_y = self.host_overlap_grads_W_h

        # print("***# component_x shape", component_x.shape)
        # print("***# component_y shape", component_y.shape)

        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_host_grad_W(self, alpha_t, beta_t):
        op_id = "mul_op_1_for_host"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.host_grad_W_h = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                           self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                          op_id), axis=0)
        return self.host_grad_W_h

    def compute_shares_for_alpha_beta_for_host_grad_b(self, global_index, op_id=None):
        op_id = "mul_op_2"
        component_x = self.host_overlap_federated_layer_grad_h
        component_y = self.host_overlap_grads_b_h
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_host_grad_b(self, alpha_t, beta_t):
        op_id = "mul_op_2"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.host_grad_b_h = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                           self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                          op_id), axis=0)
        return self.host_grad_b_h

    def receive_gradients(self, gradients):
        # receive grad_W_g and grad_b_g and reconstruct grad_W and grad_b
        grad_W = gradients[0] + self.host_grad_W_h
        grad_b = gradients[1] + self.host_grad_b_h
        # print("host receive grad_W:", grad_W)
        # print("host receive grad_b:", grad_b)
        grad_W = grad_W + self.l2_param * self.Wh
        self.localModel.apply_gradients([grad_W, grad_b])

    #
    # host side computes shares of components and gradients for guest side
    #

    def compute_shares_for_alpha_beta_for_overlap_y_2_phi_uB_2(self, global_index, op_id=None):
        op_id = "mul_op_3"
        # np.expand_dims(self.y_overlap_2 * self.phi, axis=1)
        component_x = np.expand_dims(self.y_overlap_2_phi_h, axis=1)
        component_y = self.uB_overlap_2_h
        # print("component_x shape", component_x.shape)
        # print("component_y shape", component_y.shape)
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_overlap_y_2_phi_uB_2(self, alpha_t, beta_t):
        op_id = "mul_op_3"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        overlap_y_2_phi_uB_2 = compute_matmul_share(alpha_s, alpha_t, beta_s, beta_t,
                                                    self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                        op_id))
        # overlap_y_2_phi_uB_2 has shape(len(overlap_indexes), feature_dim)
        self.overlap_y_2_phi_uB_2_h = np.squeeze(overlap_y_2_phi_uB_2, axis=1)
        return self.overlap_y_2_phi_uB_2_h

    def compute_shares_for_alpha_beta_for_overlap_y_uB(self, global_index, op_id=None):
        op_id = "mul_op_4"
        # TODO: probably do not need np.tile
        component_x = np.tile(self.half_y_overlap_h, (1, self.uB_overlap_h.shape[-1]))
        component_y = self.uB_overlap_h
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_overlap_y_uB(self, alpha_t, beta_t):
        op_id = "mul_op_4"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.overlap_y_uB_h = compute_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                     self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                         op_id))
        return self.overlap_y_uB_h

    def compute_const_part(self):
        # compute share_for_overlap_y_2_phi_uB_2_minus_y_uB
        # print("self.overlap_y_2_phi_uB_2_h:", self.overlap_y_2_phi_uB_2_h.shape)
        # print("self.overlap_y_uB_h:", self.overlap_y_uB_h.shape)

        self.const_h = compute_minus_share(np.sum(self.overlap_y_2_phi_uB_2_h, axis=0),
                                           np.sum(self.overlap_y_uB_h, axis=0))
        return self.const_h

    def compute_shares_for_alpha_beta_for_guest_non_overlap_federated_layer_grad(self, global_index, op_id=None):
        op_id = "mul_op_5"
        component_x = self.const_h
        component_y = self.ave_y_non_overlap_h
        # print("self.const_h:", self.const_h.shape)
        # print("self.ave_y_non_overlap_h:", self.ave_y_non_overlap_h.shape)
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_guest_non_overlap_federated_layer_grad(self, alpha_t, beta_t):
        op_id = "mul_op_5"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        guest_non_overlap_federated_layer_grad_h = compute_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                                          self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                                              op_id))
        self.guest_non_overlap_federated_layer_grad_h = self.alpha * guest_non_overlap_federated_layer_grad_h
        return self.guest_non_overlap_federated_layer_grad_h

    def compute_shares_for_alpha_beta_for_guest_part_overlap_federated_layer_grad(self, global_index, op_id=None):
        op_id = "mul_op_6"
        component_x = self.const_h
        component_y = self.ave_y_overlap_h
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_guest_part_overlap_federated_layer_grad(self, alpha_t, beta_t):
        op_id = "mul_op_6"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.guest_part_overlap_federated_layer_grad_h = compute_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                                                self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                                                    op_id))
        return self.guest_part_overlap_federated_layer_grad_h

    def compute_share_for_guest_overlap_federated_layer_grad(self):
        self.guest_overlap_federated_layer_grad_h = compute_add_share(self.alpha * self.guest_part_overlap_federated_layer_grad_h,
                                                                      self.gamma * self.mapping_comp_B_h)
        return self.guest_overlap_federated_layer_grad_h

    def compute_shares_for_alpha_beta_for_guest_grad_W(self, guest_overlap_grads_W_h, global_index, op_id=None):
        op_id = "mul_op_1_for_guest"
        self.guest_federated_layer_grad_h = np.zeros((len(self.ave_y_non_overlap_h) + len(self.ave_y_overlap_h), self.uB_overlap.shape[1]))
        # print("self.guest_federated_layer_grad_h shape", self.guest_federated_layer_grad_h.shape)
        # print("self.guest_non_overlap_federated_layer_grad_h shape", self.guest_non_overlap_federated_layer_grad_h.shape)
        # print("self.guest_overlap_federated_layer_grad_h shape", self.guest_overlap_federated_layer_grad_h.shape)
        self.guest_federated_layer_grad_h[self.guest_non_overlap_indexes, :] = self.guest_non_overlap_federated_layer_grad_h
        self.guest_federated_layer_grad_h[self.overlap_indexes, :] = self.guest_overlap_federated_layer_grad_h

        guest_federated_layer_grad_h = np.expand_dims(self.guest_federated_layer_grad_h, axis=1)
        # TODO: probably do not need to do this. The numpy's broadcasting will do this implicitly
        # overlap_federated_layer_grad_h = np.broadcast_to(overlap_federated_layer_grad_h, (
        # self.overlap_grads_W_h.shape[0], self.overlap_grads_W_h.shape[1], self.overlap_grads_W_h.shape[2]))
        # print("*** self.guest_federated_layer_grad_h", guest_federated_layer_grad_h.shape)
        # print("*** self.guest_overlap_grads_W_h", guest_overlap_grads_W_h.shape)
        component_x = guest_federated_layer_grad_h
        component_y = guest_overlap_grads_W_h
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_guest_grad_W(self, alpha_t, beta_t):
        op_id = "mul_op_1_for_guest"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.guest_grad_W_h = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                            self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                                op_id), axis=0)
        return self.guest_grad_W_h

    def compute_shares_for_alpha_beta_for_guest_grad_b(self, guest_overlap_grads_b_h, global_index, op_id=None):
        op_id = "mul_op_8"
        component_x = self.guest_federated_layer_grad_h
        component_y = guest_overlap_grads_b_h
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_guest_grad_b(self, alpha_t, beta_t):
        op_id = "mul_op_8"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.guest_grad_b_h = compute_sum_of_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                                            self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                                op_id), axis=0)
        return self.guest_grad_b_h

    #
    # host side computes share of loss
    #

    def compute_shares_for_alpha_beta_for_overlap_y_phi_uB(self, global_index, op_id=None):
        op_id = "mul_op_9"
        component_x = np.expand_dims(self.y_overlap_phi_h, axis=1)
        component_y = np.expand_dims(self.uB_overlap_h, axis=2)
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_overlap_y_phi_uB(self, alpha_t, beta_t):
        op_id = "mul_op_9"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.overlap_y_phi_uB_h = compute_matmul_share(alpha_s, alpha_t, beta_s, beta_t,
                                                       self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                           op_id))
        return self.overlap_y_phi_uB_h

    # enc_phi_uB_2_Phi = encrypt_matmul_2_ob(encrypt_matmul_2_ob(phi, enc_uB_2), phi.transpose())
    def compute_shares_for_alpha_beta_for_phi_overlap_uB_2(self, global_index, op_id=None):
        op_id = "mul_op_10"
        uB_overlap_2_h = np.sum(self.uB_overlap_2_h, axis=0)
        # phi has shape (1, hidden_dim)
        # uB_overlap_2_g has shape (hidden_dim, hidden_dim)
        component_x = self.phi_h
        component_y = uB_overlap_2_h
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_phi_overlap_uB_2(self, alpha_t, beta_t):
        op_id = "mul_op_10"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.phi_overlap_uB_2_h = compute_matmul_share(alpha_s, alpha_t, beta_s, beta_t,
                                                       self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                           op_id))
        return self.phi_overlap_uB_2_h

    def compute_shares_for_alpha_beta_for_phi_overlap_uB_2_phi(self, global_index, op_id=None):
        op_id = "mul_op_11"
        # phi_overlap_uB_2 has shape (1, hidden_dim)
        # phi.transpose() has shape (hidden_dim, 1)
        component_x = self.phi_overlap_uB_2_h
        component_y = self.phi_h.transpose()
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_phi_overlap_uB_2_phi(self, alpha_t, beta_t):
        op_id = "mul_op_11"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.phi_overlap_uB_2_phi_h = compute_matmul_share(alpha_s, alpha_t, beta_s, beta_t,
                                                           self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                               op_id))
        return self.phi_overlap_uB_2_phi_h

    # uA_overlap = - self.uA_overlap / self.feature_dim
    # loss_overlap = np.sum(uA_overlap * self.uB_overlap)
    def compute_shares_for_alpha_beta_for_uA_uB(self, global_index, op_id=None):
        op_id = "mul_op_12"
        uA_overlap_h = - self.uA_overlap_h / self.feature_dim
        # uA_overlap_g has shape (N_AB, hidden_dim)
        # uB_overlap_g has shape (N_AB, hidden_dim)
        component_x = uA_overlap_h
        component_y = self.uB_overlap_h
        return self.alpha_beta_computer.compute_shares_for_alpha_beta_for_mul_op(global_index, op_id,
                                                                                 component_x, component_y)

    def compute_share_for_uA_uB(self, alpha_t, beta_t):
        op_id = "mul_op_12"
        alpha_s, beta_s = self.alpha_beta_computer.get_shares_for_mul_op_alpha_beta(op_id)
        self.uA_uB_h = compute_multiply_share(alpha_s, alpha_t, beta_s, beta_t,
                                              self.alpha_beta_computer.get_shares_for_mul_op_beaver_triple(
                                                  op_id))
        return self.uA_uB_h

    def compute_share_for_loss(self):
        # print("-"*10)
        # print("self.overlap_y_phi_uB_h", np.sum(self.overlap_y_phi_uB_h))
        # print("self.phi_overlap_uB_2_phi_h", np.sum(self.phi_overlap_uB_2_phi_h))
        # print("self.uA_uB_h", np.sum(self.uA_uB_h))
        # print("len(self.ave_y_overlap_h) * np.log(2)", len(self.ave_y_overlap_h) * np.log(2))

        const = len(self.ave_y_overlap_h) * np.log(2)
        loss_h = np.sum(self.overlap_y_phi_uB_h) + 1.0 / 8 * np.sum(self.phi_overlap_uB_2_phi_h) + np.sum(
            self.uA_uB_h) + const
        # print("self.loss_h", loss_h)
        self.loss_h = loss_h + 0.5 * self.l2_param * np.sum(np.square(self.Wh))
        return self.loss_h, np.sum(self.overlap_y_phi_uB_h), np.sum(self.phi_overlap_uB_2_phi_h), np.sum(
            self.uA_uB_h), const

    def predict(self, X):
        return self.localModel.transform(X)


class LocalSecureSharingFederatedTransferLearning(object):

    def __init__(self, *, guest: SecureSharingFTLGuestModel, host: SecureSharingFTLHostModel):
        super(LocalSecureSharingFederatedTransferLearning, self).__init__()
        self.guest = guest
        self.host = host

    def set_party_A(self, party_A):
        self.party_A = party_A

    def set_party_B(self, party_B):
        self.party_B = party_B

    def create_beaver_triples(self):
        pass

    def fit(self, X_A, X_B, y, overlap_indexes, guest_non_overlap_indexes, global_index):
        self.guest.set_batch(X_A, y, guest_non_overlap_indexes=guest_non_overlap_indexes, overlap_indexes=overlap_indexes)
        self.host.set_batch(X_B, guest_non_overlap_indexes=guest_non_overlap_indexes, overlap_indexes=overlap_indexes)

        comp_B = self.host.send_components()
        comp_A = self.guest.send_components()

        self.host.receive_components(comp_A)
        self.guest.receive_components(comp_B)

        # self.party_A.set_batch(X_A, y, guest_non_overlap_indexes, overlap_indexes)
        # self.party_B.set_batch(X_B, overlap_indexes)
        # components_A = self.party_A.send_components()
        # components_B = self.party_B.send_components()

        # actual_host_grads_W, actual_host_grads_b = self.party_B.localModel.compute_gradients(X_B[overlap_indexes])

        # self.party_A.receive_components(components_B)
        # self.party_B.receive_components(components_A)

        #
        #
        #

        alpha_h, beta_h = self.host.compute_shares_for_alpha_beta_for_overlap_uB_y_2_phi_2(global_index)
        alpha_g, beta_g = self.guest.compute_shares_for_alpha_beta_for_overlap_uB_y_2_phi_2(global_index)

        overlap_uB_y_2_phi_2_h = self.host.compute_share_for_overlap_uB_y_2_phi_2(alpha_g, beta_g)
        overlap_uB_y_2_phi_2_g = self.guest.compute_share_for_overlap_uB_y_2_phi_2(alpha_h, beta_h)

        # actual_overlap_uB_y_2_phi_2 = self.party_B.overlap_uB_y_overlap_2_phi_2
        # overlap_uB_y_2_phi_2 = overlap_uB_y_2_phi_2_g + overlap_uB_y_2_phi_2_h
        # print("overlap_uB_y_2_phi_2 shape \n", overlap_uB_y_2_phi_2, overlap_uB_y_2_phi_2.shape)
        # print("actual_overlap_uB_y_2_phi_2 shape \n", actual_overlap_uB_y_2_phi_2, actual_overlap_uB_y_2_phi_2.shape)
        # assert_matrix(overlap_uB_y_2_phi_2, actual_overlap_uB_y_2_phi_2)

        overlap_federated_layer_grad_h = self.host.compute_share_for_host_overlap_federated_layer_grad()
        overlap_federated_layer_grad_g = self.guest.compute_share_for_host_overlap_federated_layer_grad()

        # actual_overlap_federated_layer_grad = self.party_B.get_loss_grads()
        # overlap_federated_layer_grad = overlap_federated_layer_grad_g + overlap_federated_layer_grad_h
        # print("overlap_federated_layer_grad shape \n", overlap_federated_layer_grad, overlap_federated_layer_grad.shape)
        # print("actual_overlap_federated_layer_grad shape \n", actual_overlap_federated_layer_grad,
        #       actual_overlap_federated_layer_grad.shape)
        # assert_matrix(overlap_federated_layer_grad, actual_overlap_federated_layer_grad)

        overlap_host_grads_W_g, overlap_host_grads_b_g = self.host.compute_shares_for_host_local_gradients()

        alpha_host_grad_W_h, beta_host_grad_W_h = self.host.compute_shares_for_alpha_beta_for_host_grad_W(global_index)
        alpha_host_grad_W_g, beta_host_grad_W_g = self.guest.compute_shares_for_alpha_beta_for_host_grad_W(overlap_host_grads_W_g, global_index)

        host_grad_W_h = self.host.compute_share_for_host_grad_W(alpha_host_grad_W_g, beta_host_grad_W_g)
        host_grad_W_g = self.guest.compute_share_for_host_grad_W(alpha_host_grad_W_h, beta_host_grad_W_h)

        alpha_host_grad_b_h, beta_host_grad_b_h = self.host.compute_shares_for_alpha_beta_for_host_grad_b(global_index)
        alpha_host_grad_b_g, beta_host_grad_b_g = self.guest.compute_shares_for_alpha_beta_for_host_grad_b(overlap_host_grads_b_g, global_index)

        host_grad_b_h = self.host.compute_share_for_host_grad_b(alpha_host_grad_b_g, beta_host_grad_b_g)
        host_grad_b_g = self.guest.compute_share_for_host_grad_b(alpha_host_grad_b_h, beta_host_grad_b_h)

        # host_loss_grad_W = host_grad_W_h + host_grad_W_g
        # host_loss_grad_b = host_grad_b_h + host_grad_b_g
        #
        # actual_host_fl_grads = self.party_B.get_loss_grads()
        #
        # # actual_grads_W, actual_grads_b = self.party_B.localModel.compute_gradients(X_B[overlap_indexes])
        # actual_host_fl_grads_ex = np.expand_dims(actual_host_fl_grads, axis=1)
        #
        # print("actual_host_fl_grads_ex.shape", actual_host_fl_grads_ex.shape)
        # print("actual_host_grads_W.shape, actual_host_grads_b.shape", actual_host_grads_W.shape, actual_host_grads_b.shape)
        # print("host_loss_grad_W.shape, host_loss_grad_b.shape", host_loss_grad_W.shape, host_loss_grad_b.shape)
        #
        # actual_loss_grads_W = np.sum(actual_host_fl_grads_ex * actual_host_grads_W, axis=0)
        # actual_loss_grads_b = np.sum(actual_host_fl_grads * actual_host_grads_b, axis=0)
        #
        # print("actual_loss_grads_W.shape, actual_loss_grads_b.shape \n", actual_loss_grads_W.shape, actual_loss_grads_b.shape)
        # print("host_loss_grad_W:", host_loss_grad_W)
        # print("actual_loss_grads_W:", actual_loss_grads_W)
        # assert_matrix(host_loss_grad_W, actual_loss_grads_W)
        # assert_matrix(host_loss_grad_b, actual_loss_grads_b)

        self.host.receive_gradients([host_grad_W_g, host_grad_b_g])

        # host_grads_W, host_grads_b = self.host.localModel.compute_gradients(X_B[overlap_indexes])
        # actual_host_grads_W, actual_host_grads_b = self.party_B.localModel.compute_gradients(X_B[overlap_indexes])
        # print("host_grads_W:", host_grads_W)
        # print("actual_host_grads_W:", actual_host_grads_W)
        # assert_matrix(host_grads_W, actual_host_grads_W)
        # assert_matrix(host_grads_b, actual_host_grads_b)


        #
        #
        #

        alpha_h, beta_h = self.host.compute_shares_for_alpha_beta_for_overlap_y_2_phi_uB_2(global_index)
        alpha_g, beta_g = self.guest.compute_shares_for_alpha_beta_for_overlap_y_2_phi_uB_2(global_index)

        overlap_y_2_phi_uB_2_h = self.host.compute_share_for_overlap_y_2_phi_uB_2(alpha_g, beta_g)
        overlap_y_2_phi_uB_2_g = self.guest.compute_share_for_overlap_y_2_phi_uB_2(alpha_h, beta_h)

        alpha_h, beta_h = self.host.compute_shares_for_alpha_beta_for_overlap_y_uB(global_index)
        alpha_g, beta_g = self.guest.compute_shares_for_alpha_beta_for_overlap_y_uB(global_index)

        overlap_y_uB_h = self.host.compute_share_for_overlap_y_uB(alpha_g, beta_g)
        overlap_y_uB_g = self.guest.compute_share_for_overlap_y_uB(alpha_h, beta_h)

        const_h = self.host.compute_const_part()
        const_g = self.guest.compute_const_part()

        alpha_h, beta_h = self.host.compute_shares_for_alpha_beta_for_guest_non_overlap_federated_layer_grad(global_index)
        alpha_g, beta_g = self.guest.compute_shares_for_alpha_beta_for_guest_non_overlap_federated_layer_grad(global_index)

        guest_non_overlap_federated_layer_grad_h = self.host.compute_share_for_guest_non_overlap_federated_layer_grad(alpha_g, beta_g)
        guest_non_overlap_federated_layer_grad_g = self.guest.compute_share_for_guest_non_overlap_federated_layer_grad(alpha_h, beta_h)

        alpha_h, beta_h = self.host.compute_shares_for_alpha_beta_for_guest_part_overlap_federated_layer_grad(global_index)
        alpha_g, beta_g = self.guest.compute_shares_for_alpha_beta_for_guest_part_overlap_federated_layer_grad(global_index)

        self.host.compute_share_for_guest_part_overlap_federated_layer_grad(alpha_g, beta_g)
        self.guest.compute_share_for_guest_part_overlap_federated_layer_grad(alpha_h, beta_h)

        guest_overlap_federated_layer_grad_h = self.host.compute_share_for_guest_overlap_federated_layer_grad()
        guest_overlap_federated_layer_grad_g = self.guest.compute_share_for_guest_overlap_federated_layer_grad()

        guest_overlap_grads_W_h, guest_overlap_grads_b_h = self.guest.compute_shares_for_guest_local_gradients()

        alpha_h, beta_h = self.host.compute_shares_for_alpha_beta_for_guest_grad_W(guest_overlap_grads_W_h, global_index)
        alpha_g, beta_g = self.guest.compute_shares_for_alpha_beta_for_guest_grad_W(global_index)

        guest_grad_W_h = self.host.compute_share_for_guest_grad_W(alpha_g, beta_g)
        guest_grad_W_g = self.guest.compute_share_for_guest_grad_W(alpha_h, beta_h)

        alpha_h, beta_h = self.host.compute_shares_for_alpha_beta_for_guest_grad_b(guest_overlap_grads_b_h, global_index)
        alpha_g, beta_g = self.guest.compute_shares_for_alpha_beta_for_guest_grad_b(global_index)

        guest_grad_b_h = self.host.compute_share_for_guest_grad_b(alpha_g, beta_g)
        guest_grad_b_g = self.guest.compute_share_for_guest_grad_b(alpha_h, beta_h)

        # guest_grad_W = guest_grad_W_g + guest_grad_W_h
        # guest_grad_b = guest_grad_b_g + guest_grad_b_h
        #
        # self.guest.receive_gradients([guest_grad_W, guest_grad_b])

        self.guest.receive_gradients([guest_grad_W_h, guest_grad_b_h])

        #
        # compute loss
        #

        alpha_h, beta_h = self.host.compute_shares_for_alpha_beta_for_overlap_y_phi_uB(global_index)
        alpha_g, beta_g = self.guest.compute_shares_for_alpha_beta_for_overlap_y_phi_uB(global_index)

        self.host.compute_share_for_overlap_y_phi_uB(alpha_g, beta_g)
        self.guest.compute_share_for_overlap_y_phi_uB(alpha_h, beta_h)

        alpha_h, beta_h = self.host.compute_shares_for_alpha_beta_for_phi_overlap_uB_2(global_index)
        alpha_g, beta_g = self.guest.compute_shares_for_alpha_beta_for_phi_overlap_uB_2(global_index)

        self.host.compute_share_for_phi_overlap_uB_2(alpha_g, beta_g)
        self.guest.compute_share_for_phi_overlap_uB_2(alpha_h, beta_h)

        alpha_h, beta_h = self.host.compute_shares_for_alpha_beta_for_phi_overlap_uB_2_phi(global_index)
        alpha_g, beta_g = self.guest.compute_shares_for_alpha_beta_for_phi_overlap_uB_2_phi(global_index)

        self.host.compute_share_for_phi_overlap_uB_2_phi(alpha_g, beta_g)
        self.guest.compute_share_for_phi_overlap_uB_2_phi(alpha_h, beta_h)

        alpha_h, beta_h = self.host.compute_shares_for_alpha_beta_for_uA_uB(global_index)
        alpha_g, beta_g = self.guest.compute_shares_for_alpha_beta_for_uA_uB(global_index)

        self.host.compute_share_for_uA_uB(alpha_g, beta_g)
        self.guest.compute_share_for_uA_uB(alpha_h, beta_h)
        loss_h, v1_h, v2_h, v3_h, const = self.host.compute_share_for_loss()
        loss_g, v1_g, v2_g, v3_g = self.guest.compute_share_for_loss()
        loss = loss_g + loss_h
        v1 = v1_h + v1_g
        v2 = v2_h + v2_g
        v3 = v3_h + v3_g

        return loss, v1, v2, v3

        # v0 = 1
        # v1 = 2
        # v2 = 3
        # v3 = 4
        # return v0, v1, v2, v3

    def predict(self, X_B):
        msg = self.host.predict(X_B)
        return self.guest.predict(msg)
