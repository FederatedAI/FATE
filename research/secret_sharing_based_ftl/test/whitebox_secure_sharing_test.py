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

from research.beaver_triples_generation.beaver_triple import fill_beaver_triple_shape, create_beaver_triples
from federatedml.ftl.plain_ftl import PlainFTLGuestModel, PlainFTLHostModel
from research.secret_sharing_based_ftl.secure_sharing_ftl import SecureSharingFTLGuestModel, SecureSharingFTLHostModel
from federatedml.ftl.test.mock_models import MockAutoencoder, MockFTLModelParam
from federatedml.ftl.test.util import assert_matrix


def create_mul_op_def():
    ops = []
    mul_op_def = dict()
    op_id = "mul_op_0"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 1, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5, 5)
    mul_op_def[op_id]["is_constant"] = True
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
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["batch_size"] = 2
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_2"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["batch_size"] = 2
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_3"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 1, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5, 5)
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["batch_size"] = 2
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_4"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["batch_size"] = 2
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_5"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["batch_size"] = 2
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_6"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["batch_size"] = 2
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_7"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (4, 5, 5)
    mul_op_def[op_id]["Y_shape"] = (4, 5, 5)
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["batch_size"] = 2
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_8"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (4, 5)
    mul_op_def[op_id]["Y_shape"] = (4, 5)
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["batch_size"] = 2
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    return mul_op_def, ops


def create_mul_op_def_2():
    ops = []
    mul_op_def = dict()

    op_id = "mul_op_0"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 1, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    # op_id = "mul_op_1"
    # mul_op_def[op_id] = dict()
    # mul_op_def[op_id]["X_shape"] = (2, 5, 5)
    # mul_op_def[op_id]["Y_shape"] = (2, 5, 5)
    # mul_op_def[op_id]["batch_size"] = 0
    # mul_op_def[op_id]["is_constant"] = True
    # mul_op_def[op_id]["mul_type"] = "multiply"
    # mul_op_def[op_id]["batch_axis"] = 0
    # ops.append(op_id)

    op_id = "mul_op_1_for_guest"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (7, 8, 5)
    mul_op_def[op_id]["Y_shape"] = (7, 8, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_1_for_host"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 8, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 8, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_2"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_3"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 1, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_4"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_5"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (5, 5)
    mul_op_def[op_id]["Y_shape"] = (5, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_6"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    # op_id = "mul_op_7"
    # mul_op_def[op_id] = dict()
    # mul_op_def[op_id]["X_shape"] = (4, 5, 5)
    # mul_op_def[op_id]["Y_shape"] = (4, 5, 5)
    # mul_op_def[op_id]["batch_size"] = 0
    # mul_op_def[op_id]["is_constant"] = True
    # mul_op_def[op_id]["mul_type"] = "multiply"
    # mul_op_def[op_id]["batch_axis"] = 0
    # ops.append(op_id)

    op_id = "mul_op_8"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (7, 5)
    mul_op_def[op_id]["Y_shape"] = (7, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_9"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 1, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5, 1)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_10"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (1, 5)
    mul_op_def[op_id]["Y_shape"] = (5, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_11"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (1, 5)
    mul_op_def[op_id]["Y_shape"] = (5, 1)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_12"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    return mul_op_def, ops


def create_mul_op_def_3():
    ops = []
    mul_op_def = dict()

    op_id = "mul_op_0"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 1, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    # op_id = "mul_op_1"
    # mul_op_def[op_id] = dict()
    # mul_op_def[op_id]["X_shape"] = (2, 5, 5)
    # mul_op_def[op_id]["Y_shape"] = (2, 5, 5)
    # mul_op_def[op_id]["batch_size"] = 0
    # mul_op_def[op_id]["is_constant"] = True
    # mul_op_def[op_id]["mul_type"] = "multiply"
    # mul_op_def[op_id]["batch_axis"] = 0
    # ops.append(op_id)

    op_id = "mul_op_1_for_guest"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (7, 8, 5)
    mul_op_def[op_id]["Y_shape"] = (7, 8, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_1_for_host"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 8, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 8, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_2"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_3"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 1, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_4"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_5"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (5, 5)
    mul_op_def[op_id]["Y_shape"] = (5, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_6"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    # op_id = "mul_op_7"
    # mul_op_def[op_id] = dict()
    # mul_op_def[op_id]["X_shape"] = (4, 5, 5)
    # mul_op_def[op_id]["Y_shape"] = (4, 5, 5)
    # mul_op_def[op_id]["batch_size"] = 0
    # mul_op_def[op_id]["is_constant"] = True
    # mul_op_def[op_id]["mul_type"] = "multiply"
    # mul_op_def[op_id]["batch_axis"] = 0
    # ops.append(op_id)

    op_id = "mul_op_8"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (4, 5)
    mul_op_def[op_id]["Y_shape"] = (4, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_9"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 1, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5, 1)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_10"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (1, 5)
    mul_op_def[op_id]["Y_shape"] = (5, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_11"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (1, 5)
    mul_op_def[op_id]["Y_shape"] = (5, 1)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_12"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (2, 5)
    mul_op_def[op_id]["Y_shape"] = (2, 5)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
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
                                             is_constant=val["is_constant"],
                                             batch_axis=val["batch_axis"])
        print("num_batch", num_batch)

    num_epoch = 1
    global_iters = num_batch * num_epoch
    party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=global_iters, num_batch=num_batch)
    return party_a_bt_map, party_b_bt_map


class TestPlainGradients(unittest.TestCase):

    # def test_generate_beaver_triple_test(self):
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

    # def test_party_a_gradient_checking_test(self):
    #     mul_op_def, ops = create_mul_op_def_2()
    #     party_a_bt_map, party_b_bt_map = generate_beaver_triples(mul_op_def)
    #
    #     U_A = np.array([[1, 2, 3, 4, 5],
    #                     [4, 5, 6, 7, 8],
    #                     [7, 8, 9, 10, 11],
    #                     [4, 5, 6, 7, 8],
    #                     [1, 5, 8, 9, 8],
    #                     [3, 4, 9, 1, 2],
    #                     [2, 7, 9, 2, 1]])
    #     U_B = np.array([[4, 2, 3, 1, 2],
    #                     [6, 5, 1, 4, 5],
    #                     [7, 4, 1, 9, 10],
    #                     [6, 5, 2, 4, 5],
    #                     [4, 1, 6, 4, 3]])
    #     y = np.array([[1], [-1], [1], [-1], [1], [1], [-1]])
    #
    #     overlap_indexes = [1, 2]
    #     guest_non_overlap_indexes = [0, 3, 4, 5, 6]
    #     host_non_overlap_indexes = [3, 4]
    #
    #     Wh = np.ones((8, U_A.shape[1]))
    #     bh = np.ones(U_A.shape[1])
    #
    #     autoencoderA = MockAutoencoder(0)
    #     autoencoderA.build(U_A.shape[1], Wh, bh)
    #     autoencoderB = MockAutoencoder(1)
    #     autoencoderB.build(U_B.shape[1], Wh, bh)
    #
    #     autoencoder_guest = MockAutoencoder(0)
    #     autoencoder_guest.build(U_A.shape[1], Wh, bh)
    #     autoencoder_host = MockAutoencoder(1)
    #     autoencoder_host.build(U_B.shape[1], Wh, bh)
    #
    #     mock_model_param = MockFTLModelParam(alpha=1, gamma=1)
    #
    #     # plain version of FTL algorithm for testing purpose
    #     partyA = PlainFTLGuestModel(autoencoderA, mock_model_param)
    #     partyB = PlainFTLHostModel(autoencoderB, mock_model_param)
    #
    #     guest = SecureSharingFTLGuestModel(autoencoder_guest, mock_model_param)
    #     host = SecureSharingFTLHostModel(autoencoder_host, mock_model_param)
    #     guest.set_bt_map(party_a_bt_map)
    #     host.set_bt_map(party_b_bt_map)
    #
    #     for global_index in range(len(party_a_bt_map)):
    #         print("global index: ", global_index)
    #         guest.set_batch(U_A,
    #                         y,
    #                         overlap_indexes=overlap_indexes,
    #                         guest_non_overlap_indexes=guest_non_overlap_indexes,
    #                         host_non_overlap_indexes=host_non_overlap_indexes)
    #         host.set_batch(U_B,
    #                        overlap_indexes=overlap_indexes,
    #                        guest_non_overlap_indexes=guest_non_overlap_indexes,
    #                        host_non_overlap_indexes=host_non_overlap_indexes)
    #
    #         partyA.set_batch(U_A, y, guest_non_overlap_indexes, overlap_indexes)
    #         partyB.set_batch(U_B, overlap_indexes)
    #
    #         comp_B = host.send_components()
    #         comp_A = guest.send_components()
    #
    #         host.receive_components(comp_A)
    #         guest.receive_components(comp_B)
    #
    #         comp_A_beta1, comp_A_beta2, mapping_comp_A = partyA.send_components()
    #         U_B_overlap, U_B_overlap_2, mapping_comp_B = partyB.send_components()
    #
    #         partyA.receive_components([U_B_overlap, U_B_overlap_2, mapping_comp_B])
    #         partyB.receive_components([comp_A_beta1, comp_A_beta2, mapping_comp_A])
    #
    #         alpha_h, beta_h = host.compute_shares_for_alpha_beta_for_overlap_y_2_phi_uB_2(global_index)
    #         alpha_g, beta_g = guest.compute_shares_for_alpha_beta_for_overlap_y_2_phi_uB_2(global_index)
    #
    #         overlap_y_2_phi_uB_2_h = host.compute_share_for_overlap_y_2_phi_uB_2(alpha_g, beta_g)
    #         overlap_y_2_phi_uB_2_g = guest.compute_share_for_overlap_y_2_phi_uB_2(alpha_h, beta_h)
    #
    #         overlap_y_2_phi_uB_2 = overlap_y_2_phi_uB_2_g + overlap_y_2_phi_uB_2_h
    #         actual_overlap_y_2_phi_uB_2 = partyA.overlap_y_2_phi_uB_2
    #
    #         print("overlap_y_2_phi_uB_2 shape \n", overlap_y_2_phi_uB_2, overlap_y_2_phi_uB_2.shape)
    #         print("actual_overlap_uB_y_2_phi_2 shape \n", actual_overlap_y_2_phi_uB_2, actual_overlap_y_2_phi_uB_2.shape)
    #         assert_matrix(overlap_y_2_phi_uB_2, actual_overlap_y_2_phi_uB_2)
    #
    #         alpha_h, beta_h = host.compute_shares_for_alpha_beta_for_overlap_y_uB(global_index)
    #         alpha_g, beta_g = guest.compute_shares_for_alpha_beta_for_overlap_y_uB(global_index)
    #
    #         overlap_y_uB_h = host.compute_share_for_overlap_y_uB(alpha_g, beta_g)
    #         overlap_y_uB_g = guest.compute_share_for_overlap_y_uB(alpha_h, beta_h)
    #
    #         overlap_y_uB = overlap_y_uB_g + overlap_y_uB_h
    #         actual_overlap_y_uB = partyA.overlap_y_uB
    #
    #         print("overlap_y_uB shape \n", overlap_y_uB, overlap_y_uB.shape)
    #         print("actual_overlap_y_uB shape \n", actual_overlap_y_uB, actual_overlap_y_uB.shape)
    #         assert_matrix(overlap_y_uB, actual_overlap_y_uB)
    #
    #         const_h = host.compute_const_part()
    #         const_g = guest.compute_const_part()
    #
    #         const = const_g + const_h
    #         actual_const = partyA.const
    #         print("const shape \n", const, const.shape)
    #         print("actual_const shape \n", actual_const, actual_const.shape)
    #         assert_matrix(const, actual_const)
    #
    #         alpha_h, beta_h = host.compute_shares_for_alpha_beta_for_guest_non_overlap_federated_layer_grad(global_index)
    #         alpha_g, beta_g = guest.compute_shares_for_alpha_beta_for_guest_non_overlap_federated_layer_grad(global_index)
    #
    #         guest_non_overlap_federated_layer_grad_h = host.compute_share_for_guest_non_overlap_federated_layer_grad(alpha_g, beta_g)
    #         guest_non_overlap_federated_layer_grad_g = guest.compute_share_for_guest_non_overlap_federated_layer_grad(alpha_h, beta_h)
    #         guest_non_overlap_federated_layer_grad = guest_non_overlap_federated_layer_grad_h + guest_non_overlap_federated_layer_grad_g
    #
    #         alpha_h, beta_h = host.compute_shares_for_alpha_beta_for_guest_part_overlap_federated_layer_grad(global_index)
    #         alpha_g, beta_g = guest.compute_shares_for_alpha_beta_for_guest_part_overlap_federated_layer_grad(global_index)
    #
    #         host.compute_share_for_guest_part_overlap_federated_layer_grad(alpha_g, beta_g)
    #         guest.compute_share_for_guest_part_overlap_federated_layer_grad(alpha_h, beta_h)
    #
    #         guest_overlap_federated_layer_grad_h = host.compute_share_for_guest_overlap_federated_layer_grad()
    #         guest_overlap_federated_layer_grad_g = guest.compute_share_for_guest_overlap_federated_layer_grad()
    #         guest_overlap_federated_layer_grad = guest_overlap_federated_layer_grad_h + guest_overlap_federated_layer_grad_g
    #
    #         print("guest_overlap_federated_layer_grad", guest_overlap_federated_layer_grad)
    #         print("guest_non_overlap_federated_layer_grad", guest_non_overlap_federated_layer_grad)
    #
    #         actual_guest_federated_layer_grad = partyA.loss_grads
    #
    #         guest_federated_layer_grad = np.zeros((len(y), U_A.shape[1]))
    #         guest_federated_layer_grad[guest_non_overlap_indexes, :] = guest_non_overlap_federated_layer_grad
    #         guest_federated_layer_grad[overlap_indexes, :] = guest_overlap_federated_layer_grad
    #         print("overlap_indexes:", overlap_indexes)
    #         print("guest_non_overlap_indexes:", guest_non_overlap_indexes)
    #         print("guest_federated_layer_grad shape", guest_federated_layer_grad, guest_federated_layer_grad.shape)
    #         print("actual_guest_federated_layer_grad shape", actual_guest_federated_layer_grad,
    #               actual_guest_federated_layer_grad.shape)
    #         assert_matrix(guest_federated_layer_grad, actual_guest_federated_layer_grad)
    #
    #         guest_overlap_grads_W_h, guest_overlap_grads_b_h = guest.compute_shares_for_guest_local_gradients()
    #
    #         alpha_h, beta_h = host.compute_shares_for_alpha_beta_for_guest_grad_W(guest_overlap_grads_W_h, global_index)
    #         alpha_g, beta_g = guest.compute_shares_for_alpha_beta_for_guest_grad_W(global_index)
    #
    #         guest_grad_W_h = host.compute_share_for_guest_grad_W(alpha_g, beta_g)
    #         guest_grad_W_g = guest.compute_share_for_guest_grad_W(alpha_h, beta_h)
    #         guest_grad_W = guest_grad_W_h + guest_grad_W_g
    #
    #         # grads_W, grads_b = autoencoderA.compute_gradients(U_A)
    #         actual_grads_W = np.sum(autoencoderA.get_loss_grad_W(), axis=0)
    #         actual_grads_b = np.sum(autoencoderA.get_loss_grad_b(), axis=0)
    #
    #         # actual_guest_federated_layer_grad_ex = np.expand_dims(actual_guest_federated_layer_grad, axis=1)
    #         # actual_grads_W = np.sum(np.multiply(actual_guest_federated_layer_grad_ex, grads_W), axis=0)
    #         print("guest_grad_W shape", guest_grad_W, guest_grad_W.shape)
    #         print("actual_grads_W shape", actual_grads_W, actual_grads_W.shape)
    #         assert_matrix(guest_grad_W, actual_grads_W)
    #
    #         alpha_h, beta_h = host.compute_shares_for_alpha_beta_for_guest_grad_b(guest_overlap_grads_b_h, global_index)
    #         alpha_g, beta_g = guest.compute_shares_for_alpha_beta_for_guest_grad_b(global_index)
    #
    #         guest_grad_b_h = host.compute_share_for_guest_grad_b(alpha_g, beta_g)
    #         guest_grad_b_g = guest.compute_share_for_guest_grad_b(alpha_h, beta_h)
    #         guest_grad_b = guest_grad_b_h + guest_grad_b_g
    #
    #         # actual_grads_b = np.sum(np.multiply(actual_guest_federated_layer_grad, grads_b), axis=0)
    #         print("guest_grad_b shape", guest_grad_b, guest_grad_b.shape)
    #         print("actual_grads_b shape", actual_grads_b, actual_grads_b.shape)
    #         assert_matrix(guest_grad_b, actual_grads_b)
    #
    #         alpha_h, beta_h = host.compute_shares_for_alpha_beta_for_overlap_y_phi_uB(global_index)
    #         alpha_g, beta_g = guest.compute_shares_for_alpha_beta_for_overlap_y_phi_uB(global_index)
    #
    #         host.compute_share_for_overlap_y_phi_uB(alpha_g, beta_g)
    #         guest.compute_share_for_overlap_y_phi_uB(alpha_h, beta_h)
    #
    #         alpha_h, beta_h = host.compute_shares_for_alpha_beta_for_phi_overlap_uB_2(global_index)
    #         alpha_g, beta_g = guest.compute_shares_for_alpha_beta_for_phi_overlap_uB_2(global_index)
    #
    #         host.compute_share_for_phi_overlap_uB_2(alpha_g, beta_g)
    #         guest.compute_share_for_phi_overlap_uB_2(alpha_h, beta_h)
    #
    #         alpha_h, beta_h = host.compute_shares_for_alpha_beta_for_phi_overlap_uB_2_phi(global_index)
    #         alpha_g, beta_g = guest.compute_shares_for_alpha_beta_for_phi_overlap_uB_2_phi(global_index)
    #
    #         host.compute_share_for_phi_overlap_uB_2_phi(alpha_g, beta_g)
    #         guest.compute_share_for_phi_overlap_uB_2_phi(alpha_h, beta_h)
    #
    #         alpha_h, beta_h = host.compute_shares_for_alpha_beta_for_uA_uB(global_index)
    #         alpha_g, beta_g = guest.compute_shares_for_alpha_beta_for_uA_uB(global_index)
    #
    #         host.compute_share_for_uA_uB(alpha_g, beta_g)
    #         guest.compute_share_for_uA_uB(alpha_h, beta_h)
    #         loss_h, _, _, _, _ = host.compute_share_for_loss()
    #         loss_g, _, _, _ = guest.compute_share_for_loss()
    #
    #         loss = loss_g + loss_h
    #         actual_loss = partyA.send_loss()
    #         print("expected loss", loss)
    #         print("actual loss", actual_loss)
    #         assert round(loss, 8) == round(actual_loss, 8)

    def test_party_b_gradient_checking_test(self):
        mul_op_def, ops = create_mul_op_def_3()
        party_a_bt_map, party_b_bt_map = generate_beaver_triples(mul_op_def)

        # U_A = np.array([[1, 2, 3, 4, 5],
        #                 [4, 5, 6, 7, 8],
        #                 [7, 8, 9, 10, 11],
        #                 [4, 5, 6, 7, 8]])
        # U_B = np.array([[4, 2, 3, 1, 2],
        #                 [6, 5, 1, 4, 5],
        #                 [7, 4, 1, 9, 10],
        #                 [6, 5, 1, 4, 5]])
        # y = np.array([[1], [-1], [1], [-1]])
        #
        # overlap_indexes = [1, 2]
        # non_overlap_indexes = [0, 3]
        #
        # Wh = np.ones((5, U_A.shape[1]))
        # bh = np.ones(U_A.shape[1])

        U_A = np.array([[1, 2, 3, 4, 5],
                        [4, 5, 6, 7, 8],
                        [7, 8, 9, 10, 11],
                        [4, 5, 6, 7, 8],
                        [1, 5, 8, 9, 8],
                        [3, 4, 9, 1, 2],
                        [2, 7, 9, 2, 1]])
        U_B = np.array([[4, 2, 3, 1, 2],
                        [6, 5, 1, 4, 5],
                        [7, 4, 1, 9, 10],
                        [6, 5, 2, 4, 5],
                        [4, 1, 6, 4, 3]])
        y = np.array([[1], [-1], [1], [-1], [1], [1], [-1]])

        overlap_indexes = [1, 2]
        guest_non_overlap_indexes = [0, 3, 4, 5, 6]
        host_non_overlap_indexes = [3, 4]

        Wh = np.ones((8, U_A.shape[1]))
        bh = np.ones(U_A.shape[1])

        autoencoderA = MockAutoencoder(0)
        autoencoderA.build(U_A.shape[1], Wh, bh)
        autoencoderB = MockAutoencoder(1)
        autoencoderB.build(U_B.shape[1], Wh, bh)

        mock_model_param = MockFTLModelParam(alpha=1, gamma=1)

        partyA = PlainFTLGuestModel(autoencoderA, mock_model_param)
        # partyA.set_batch(U_A, y, overlap_indexes, non_overlap_indexes)
        partyB = PlainFTLHostModel(autoencoderB, mock_model_param)
        # partyB.set_batch(U_B, overlap_indexes)

        guest = SecureSharingFTLGuestModel(autoencoderA, mock_model_param)
        host = SecureSharingFTLHostModel(autoencoderB, mock_model_param)
        guest.set_bt_map(party_a_bt_map)
        host.set_bt_map(party_b_bt_map)

        for global_index in range(len(party_a_bt_map)):
            print("global index: ", global_index)
            # guest.set_batch(U_A, y, overlap_indexes, non_overlap_indexes)
            # host.set_batch(U_B, overlap_indexes)
            guest.set_batch(U_A,
                            y,
                            overlap_indexes=overlap_indexes,
                            guest_non_overlap_indexes=guest_non_overlap_indexes)
            host.set_batch(U_B,
                           overlap_indexes=overlap_indexes,
                           guest_non_overlap_indexes=guest_non_overlap_indexes)

            # partyA.set_batch(U_A, y, non_overlap_indexes, overlap_indexes)
            # partyB.set_batch(U_B, overlap_indexes)

            partyA.set_batch(U_A, y, guest_non_overlap_indexes, overlap_indexes)
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

            overlap_federated_layer_grad_h = host.compute_share_for_host_overlap_federated_layer_grad()
            overlap_federated_layer_grad_g = guest.compute_share_for_host_overlap_federated_layer_grad()

            actual_overlap_federated_layer_grad = partyB.get_loss_grads()
            overlap_federated_layer_grad = overlap_federated_layer_grad_g + overlap_federated_layer_grad_h
            print("overlap_federated_layer_grad shape \n", overlap_federated_layer_grad, overlap_federated_layer_grad.shape)
            print("actual_overlap_federated_layer_grad shape \n", actual_overlap_federated_layer_grad, actual_overlap_federated_layer_grad.shape)
            assert_matrix(overlap_federated_layer_grad, actual_overlap_federated_layer_grad)

            overlap_grads_W_g, overlap_grads_b_g = host.compute_shares_for_host_local_gradients()

            alpha_grad_W_h, beta_grad_W_h = host.compute_shares_for_alpha_beta_for_host_grad_W(global_index)
            alpha_grad_W_g, beta_grad_W_g = guest.compute_shares_for_alpha_beta_for_host_grad_W(overlap_grads_W_g, global_index)

            print("alpha_grad_W_h, beta_grad_W_h", alpha_grad_W_h.shape, beta_grad_W_h.shape)
            print("alpha_grad_W_g, beta_grad_W_g", alpha_grad_W_g.shape, beta_grad_W_g.shape)

            grad_W_h = host.compute_share_for_host_grad_W(alpha_grad_W_g, beta_grad_W_g)
            grad_W_g = guest.compute_share_for_host_grad_W(alpha_grad_W_h, beta_grad_W_h)

            alpha_grad_b_h, beta_grad_b_h = host.compute_shares_for_alpha_beta_for_host_grad_b(global_index)
            alpha_grad_b_g, beta_grad_b_g = guest.compute_shares_for_alpha_beta_for_host_grad_b(overlap_grads_b_g, global_index)

            print("alpha_grad_b_h, beta_grad_b_h", alpha_grad_b_h.shape, beta_grad_b_h.shape)
            print("alpha_grad_b_g, beta_grad_b_g", alpha_grad_b_g.shape, beta_grad_b_g.shape)

            grad_b_h = host.compute_share_for_host_grad_b(alpha_grad_b_g, beta_grad_b_g)
            grad_b_g = guest.compute_share_for_host_grad_b(alpha_grad_b_h, beta_grad_b_h)

            grad_W = grad_W_h + grad_W_g
            grad_b = grad_b_h + grad_b_g

            print("grad_W, grad_b \n", grad_W, grad_b, grad_W.shape, grad_b.shape)
            actual_grad_W = np.sum(autoencoderB.get_loss_grad_W(), axis=0)
            actual_grad_b = np.sum(autoencoderB.get_loss_grad_b(), axis=0)
            print("actual_grad_W, actual_grad_b \n", actual_grad_W, actual_grad_b, actual_grad_W.shape, actual_grad_b.shape)
            assert_matrix(grad_W, actual_grad_W)
            assert_matrix(grad_b, actual_grad_b)


if __name__ == '__main__':
    unittest.main()
