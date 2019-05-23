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

from federatedml.ftl.test.util import assert_matrix
from research.beaver_triples_generation.beaver_triple import fill_op_beaver_triple_matrix_shape, create_beaver_triples


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
        num_batch = fill_op_beaver_triple_matrix_shape(mul_ops,
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

    def test_generate_beaver_triple_test(self):
        mul_op_def, ops = create_mul_op_def()
        party_a_bt_map, party_b_bt_map = generate_beaver_triples(mul_op_def)

        for op in ops:
            A0 = party_a_bt_map[0][op]["A0"]
            B0 = party_a_bt_map[0][op]["B0"]
            C0 = party_a_bt_map[0][op]["C0"]

            A1 = party_b_bt_map[0][op]["A1"]
            B1 = party_b_bt_map[0][op]["B1"]
            C1 = party_b_bt_map[0][op]["C1"]

            mul_type = mul_op_def[op]["mul_type"]

            A = A0 + A1
            B = B0 + B1
            C = C0 + C1
            if mul_type == "matmul":
                actual_C = np.matmul(A, B)
            else:
                actual_C = np.multiply(A, B)
            print("op: ", op)
            print("C: \n", C, C.shape)
            print("AB: \n", actual_C, actual_C.shape)
            assert C.shape == actual_C.shape
            assert_matrix(C, actual_C)
        print("test passed !")


if __name__ == '__main__':
    unittest.main()
