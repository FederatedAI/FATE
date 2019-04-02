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

import unittest

import numpy as np

from federatedml.ftl.beaver_triple import fill_beaver_triple_shape, create_beaver_triples
from federatedml.ftl.plain_ftl import PlainFTLGuestModel, PlainFTLHostModel
from federatedml.ftl.secure_sharing_ftl import SecureSharingFTLGuestModel, SecureSharingFTLHostModel
from federatedml.ftl.test.mock_models import MockAutoencoder, MockFTLModelParam
from federatedml.ftl.test.util import assert_matrix


# def run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B, y, overlap_indexes, non_overlap_indexes,
#                                public_key=None, private_key=None, is_encrypted=False):
#
#     fake_model_param = MockFTLModelParam(alpha=1)
#     if is_encrypted:
#         partyA = EncryptedFTLGuestModel(autoencoderA, fake_model_param, public_key=public_key, private_key=private_key)
#         partyA.set_batch(U_A, y, non_overlap_indexes, overlap_indexes)
#         partyB = EncryptedFTLHostModel(autoencoderB, fake_model_param, public_key=public_key, private_key=private_key)
#         partyB.set_batch(U_B, overlap_indexes)
#     else:
#         partyA = PlainFTLGuestModel(autoencoderA, fake_model_param)
#         partyA.set_batch(U_A, y, non_overlap_indexes, overlap_indexes)
#         partyB = PlainFTLHostModel(autoencoderB, fake_model_param)
#         partyB.set_batch(U_B, overlap_indexes)
#
#     comp_A_beta1, comp_A_beta2, mapping_comp_A = partyA.send_components()
#     U_B_overlap, U_B_overlap_2, mapping_comp_B = partyB.send_components()
#
#     partyA.receive_components([U_B_overlap, U_B_overlap_2, mapping_comp_B])
#     partyB.receive_components([comp_A_beta1, comp_A_beta2, mapping_comp_A])
#
#     return partyA, partyB

def create_mul_op_def():

    ops = []
    mul_op_def = dict()
    op_id = "mul_op_0"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 1, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5, 5)
    mul_op_def[op_id]["batch_size"] = 2
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    # op_id = "mul_op_0_1"
    # mul_op_def[op_id] = dict()
    # mul_op_def[op_id]["X_shape"] = (2, 5)
    # mul_op_def[op_id]["Y_shape"] = (2, 5)
    # mul_op_def[op_id]["batch_size"] = 2
    # mul_op_def[op_id]["mul_type"] = "multiply"
    # mul_op_def[op_id]["batch_axis"] = 0
    # ops.append(op_id)

    op_id = "mul_op_1"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5, 5)
    mul_op_def[op_id]["batch_size"] = 2
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_2"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["batch_size"] = 2
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)
    return mul_op_def, ops


def generate_beaver_triples(mul_op_def):

    num_batch = 1
    mul_ops = dict()
    for key, val in mul_op_def.items():
        num_batch = fill_beaver_triple_shape(mul_ops,
                                             op_id=key,
                                             X_shape=val["X_shape"],
                                             Y_shape=val["Y_shape"],
                                             batch_size=val["batch_size"],
                                             mul_type=val["mul_type"],
                                             batch_axis=val["batch_axis"])
        print("num_batch", num_batch)

    num_epoch = 1
    global_iters = num_batch * num_epoch
    party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=global_iters, num_batch=num_batch)
    return party_a_bt_map, party_b_bt_map


class TestPlainGradients(unittest.TestCase):

    # def test_generate_beaver_triple_test(self):
    #
    #     mul_op_def, ops = create_mul_op_def()
    #     party_a_bt_map, party_b_bt_map = generate_beaver_triples(mul_op_def)
    #
    #     for op in ops:
    #         A0 = party_a_bt_map[0][op]["A0"]
    #         B0 = party_a_bt_map[0][op]["B0"]
    #         C0 = party_a_bt_map[0][op]["C0"]
    #
    #         A1 = party_b_bt_map[0][op]["A1"]
    #         B1 = party_b_bt_map[0][op]["B1"]
    #         C1 = party_b_bt_map[0][op]["C1"]
    #
    #         mul_type = mul_op_def[op]["mul_type"]
    #
    #         A = A0 + A1
    #         B = B0 + B1
    #         C = C0 + C1
    #         if mul_type == "matmul":
    #             actual_C = np.matmul(A, B)
    #         else:
    #             actual_C = np.multiply(A, B)
    #         print("op: ", op)
    #         print("C: \n", C, C.shape)
    #         print("AB: \n", actual_C, actual_C.shape)
    #         assert C.shape == actual_C.shape
    #         assert_matrix(C, actual_C)
    #     print("test passed !")

    def test_party_b_gradient_checking_test(self):
        mul_op_def, ops = create_mul_op_def()
        party_a_bt_map, party_b_bt_map = generate_beaver_triples(mul_op_def)

        U_A = np.array([[1, 2, 3, 4, 5],
                        [4, 5, 6, 7, 8],
                        [7, 8, 9, 10, 11],
                        [4, 5, 6, 7, 8]])
        U_B = np.array([[4, 2, 3, 1, 2],
                        [6, 5, 1, 4, 5],
                        [7, 4, 1, 9, 10],
                        [6, 5, 1, 4, 5]])
        y = np.array([[1], [-1], [1], [-1]])

        overlap_indexes = [1, 2]
        non_overlap_indexes = [0, 3]

        Wh = np.ones((5, U_A.shape[1]))
        bh = np.zeros(U_A.shape[1])

        autoencoderA = MockAutoencoder(0)
        autoencoderA.build(U_A.shape[1], Wh, bh)
        autoencoderB = MockAutoencoder(1)
        autoencoderB.build(U_B.shape[1], Wh, bh)

        mock_model_param = MockFTLModelParam(alpha=1, gamma=1)

        partyA = PlainFTLGuestModel(autoencoderA, mock_model_param)
        # partyA.set_batch(U_A, y, non_overlap_indexes, overlap_indexes)
        partyB = PlainFTLHostModel(autoencoderB, mock_model_param)
        # partyB.set_batch(U_B, overlap_indexes)

        guest = SecureSharingFTLGuestModel(autoencoderA, mock_model_param)
        host = SecureSharingFTLHostModel(autoencoderB, mock_model_param)
        guest.set_bt_map(party_a_bt_map)
        host.set_bt_map(party_b_bt_map)

        for global_index in range(len(party_a_bt_map)):
            print("global index: ", global_index)
            guest.set_batch(U_A, y, non_overlap_indexes, overlap_indexes)
            host.set_batch(U_B, overlap_indexes)
            partyA.set_batch(U_A, y, non_overlap_indexes, overlap_indexes)
            partyB.set_batch(U_B, overlap_indexes)

            comp_B = host.send_components()
            comp_A = guest.send_components()

            host.receive_components(comp_A)
            guest.receive_components(comp_B)

            comp_A_beta1, comp_A_beta2, mapping_comp_A = partyA.send_components()
            U_B_overlap, U_B_overlap_2, mapping_comp_B = partyB.send_components()

            partyA.receive_components([U_B_overlap, U_B_overlap_2, mapping_comp_B])
            partyB.receive_components([comp_A_beta1, comp_A_beta2, mapping_comp_A])

            alpha_h, beta_h = host.compute_shares_for_alpha_beta_for_overlap_uB_y_2_phi_2(global_index)
            alpha_g, beta_g = guest.compute_shares_for_alpha_beta_for_overlap_uB_y_2_phi_2(global_index)

            overlap_uB_y_2_phi_2_h = host.compute_share_for_overlap_uB_y_2_phi_2(alpha_g, beta_g)
            overlap_uB_y_2_phi_2_g = guest.compute_share_for_overlap_uB_y_2_phi_2(alpha_h, beta_h)

            actual_overlap_uB_y_2_phi_2 = partyB.overlap_uB_y_overlap_2_phi_2
            overlap_uB_y_2_phi_2 = overlap_uB_y_2_phi_2_g + overlap_uB_y_2_phi_2_h
            print("overlap_uB_y_2_phi_2 shape \n", overlap_uB_y_2_phi_2, overlap_uB_y_2_phi_2.shape)
            print("actual_overlap_uB_y_2_phi_2 shape \n", actual_overlap_uB_y_2_phi_2, actual_overlap_uB_y_2_phi_2.shape)
            assert_matrix(overlap_uB_y_2_phi_2, actual_overlap_uB_y_2_phi_2)

            overlap_federated_layer_grad_h = host.compute_share_for_overlap_federated_layer_grad()
            overlap_federated_layer_grad_g = guest.compute_share_for_overlap_federated_layer_grad()

            actual_overlap_federated_layer_grad = partyB.get_loss_grads()
            overlap_federated_layer_grad = overlap_federated_layer_grad_g + overlap_federated_layer_grad_h
            print("overlap_federated_layer_grad shape \n", overlap_federated_layer_grad, overlap_federated_layer_grad.shape)
            print("actual_overlap_federated_layer_grad shape \n", actual_overlap_federated_layer_grad, actual_overlap_federated_layer_grad.shape)
            assert_matrix(overlap_federated_layer_grad, actual_overlap_federated_layer_grad)

            overlap_grads_W_g, overlap_grads_b_g = host.compute_shares_for_local_gradients()

            alpha_grad_W_h, beta_grad_W_h = host.compute_shares_for_alpha_beta_for_grad_W(global_index)
            alpha_grad_W_g, beta_grad_W_g = guest.compute_shares_for_alpha_beta_for_grad_W(overlap_grads_W_g,
                                                                                           global_index)

            print("alpha_grad_W_h, beta_grad_W_h", alpha_grad_W_h.shape, beta_grad_W_h.shape)
            print("alpha_grad_W_g, beta_grad_W_g", alpha_grad_W_g.shape, beta_grad_W_g.shape)

            grad_W_h = host.compute_share_for_grad_W(alpha_grad_W_g, beta_grad_W_g)
            grad_W_g = guest.compute_share_for_grad_W(alpha_grad_W_h, beta_grad_W_h)

            alpha_grad_b_h, beta_grad_b_h = host.compute_shares_for_alpha_beta_for_grad_b(global_index)
            alpha_grad_b_g, beta_grad_b_g = guest.compute_shares_for_alpha_beta_for_grad_b(overlap_grads_b_g,
                                                                                           global_index)

            grad_b_h = host.compute_share_for_grad_b(alpha_grad_b_g, beta_grad_b_g)
            grad_b_g = guest.compute_share_for_grad_b(alpha_grad_b_h, beta_grad_b_h)

            grad_W = grad_W_h + grad_W_g
            grad_b = grad_b_h + grad_b_g

            print("grad_W, grad_b \n", grad_W, grad_b, grad_W.shape, grad_b.shape)
            actual_grad_W = np.sum(autoencoderB.get_loss_grad_W(), axis=0)
            actual_grad_b = np.sum(autoencoderB.get_loss_grad_b(), axis=0)
            print("actual_grad_W, actual_grad_b \n", actual_grad_W, actual_grad_b, actual_grad_W.shape, actual_grad_b.shape)
            assert_matrix(grad_W, actual_grad_W)
            assert_matrix(grad_b, actual_grad_b)


#######################################################################################################################
    #     # partyA, partyB = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B, y, overlap_indexes, non_overlap_indexes)
    #     # loss_grads_B_1 = partyB.get_loss_grads()
    #     # loss1 = partyA.send_loss()
    #     #
    #     # U_B_prime = np.array([[4, 2, 3, 1, 2],
    #     #                       [6, 5, 1.001, 4, 5],
    #     #                       [7, 4, 1, 9, 10],
    #     #                       [6, 5, 1, 4, 5]])
    #     #
    #     # partyA, partyB = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B_prime, y, overlap_indexes, non_overlap_indexes)
    #     # loss_grads_B_2 = partyB.get_loss_grads()
    #     # loss2 = partyA.send_loss()
    #     #
    #     # grad_approx = (loss2 - loss1) / 0.001
    #     # grad_real = loss_grads_B_1[0, 2]
    #     # grad_diff = np.abs(grad_approx - grad_real)
    #     # assert grad_diff < 0.001


if __name__ == '__main__':
    unittest.main()
