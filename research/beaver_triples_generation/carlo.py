import numpy as np

from research.beaver_triples_generation.secret_sharing_ops import generate_random_3_dim_matrix, generate_random_matrix


def schema_copy(bt_map):
    _bt_map = [dict() for _ in range(len(bt_map))]
    for index, item in enumerate(bt_map):
        for key in item.keys():
            _bt_map[index][key] = dict()
    return _bt_map


def carlo_deal_data(party_a_bt_map, party_b_bt_map, mul_ops):
    # print("len(party_a_bt_map) len(party_b_bt_map)", len(party_a_bt_map), len(party_b_bt_map))
    assert len(party_a_bt_map) == len(party_b_bt_map)

    to_party_a_bt_map = schema_copy(party_a_bt_map)
    to_party_b_bt_map = schema_copy(party_b_bt_map)

    global_iters = len(party_a_bt_map)
    for i in range(global_iters):
        a_bt_i = party_a_bt_map[i]
        b_bt_i = party_b_bt_map[i]

        for op_id in mul_ops.keys():

            U0 = a_bt_i[op_id]["U0"]
            E0 = a_bt_i[op_id]["E0"]
            V1 = b_bt_i[op_id]["V1"]
            F1 = b_bt_i[op_id]["F1"]

            mul_type = mul_ops[op_id]["mul_type"]
            if mul_type == "matmul":
                S = np.matmul(U0, V1)
                K = np.matmul(F1, E0)
            elif mul_type == "multiply":
                S = np.multiply(U0, V1)
                K = np.multiply(F1, E0)
            else:
                raise TypeError("does not support" + mul_type)

            print(op_id, "S", S.shape, "K", K.shape)
            if len(S.shape) == 2:
                S0 = generate_random_matrix(S.shape[0], S.shape[1])
                K0 = generate_random_matrix(K.shape[0], K.shape[1])
            elif len(S.shape) == 3:
                S0 = generate_random_3_dim_matrix(S.shape[0], S.shape[1], S.shape[2])
                K0 = generate_random_3_dim_matrix(K.shape[0], K.shape[1], K.shape[2])
            else:
                raise TypeError("does not support shape {0}".format(len(S.shape)))
            S1 = S - S0
            K1 = K - K0

            to_party_a_bt_map[i][op_id]["S0"] = S0
            to_party_b_bt_map[i][op_id]["S1"] = S1

            to_party_a_bt_map[i][op_id]["K0"] = K0
            to_party_b_bt_map[i][op_id]["K1"] = K1

    # TODO: sends _party_a_bt_map to party A and _party_b_bt_map to party B
    return to_party_a_bt_map, to_party_b_bt_map
