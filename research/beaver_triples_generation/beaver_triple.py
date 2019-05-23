import numpy as np

from research.beaver_triples_generation.carlo import carlo_deal_data
from research.beaver_triples_generation.secret_sharing_ops import share, generate_random_matrix, \
    generate_random_3_dim_matrix


def get_matrix_shapes(val, global_iter_index, num_batch):
    if val["is_constant"]:
        if val["num_dim"] == 2:
            k = val["left_0"]
            ll = val["left_1"]
            p = val["right_0"]
            q = val["right_1"]
            return k, ll, p, q
        elif val["num_dim"] == 3:
            j = val["left_0"]
            k = val["left_1"]
            ll = val["left_2"]
            o = val["right_0"]
            p = val["right_1"]
            q = val["right_2"]
            return j, k, ll, o, p, q
        else:
            raise TypeError("currently does not support {} num of dimensions".format(val["num_dim"]))

    if (global_iter_index + 1) % num_batch == 0:
        # if last batch
        if val["num_dim"] == 2:
            k = val["last_left_0"]
            ll = val["last_left_1"]
            p = val["last_right_0"]
            q = val["last_right_1"]
            return k, ll, p, q
        elif val["num_dim"] == 3:
            j = val["last_left_0"]
            k = val["last_left_1"]
            ll = val["last_left_2"]
            o = val["last_right_0"]
            p = val["last_right_1"]
            q = val["last_right_2"]
            return j, k, ll, o, p, q
        else:
            raise TypeError("currently does not support {} num of dimensions".format(val["num_dim"]))
    else:
        if val["num_dim"] == 2:
            k = val["left_0"]
            ll = val["left_1"]
            p = val["right_0"]
            q = val["right_1"]
            return k, ll, p, q
        elif val["num_dim"] == 3:
            j = val["left_0"]
            k = val["left_1"]
            ll = val["left_2"]
            o = val["right_0"]
            p = val["right_1"]
            q = val["right_2"]
            return j, k, ll, o, p, q
        else:
            raise TypeError("currently does not support {} num of dimensions".format(val["num_dim"]))


def fill_beaver_triple_matrix_shape(mul_op_def, num_epoch=1):
    num_batch = 1
    mul_ops = dict()
    for op_id, attr in mul_op_def.items():
        num_batch = fill_op_beaver_triple_matrix_shape(mul_ops,
                                                       op_id=op_id,
                                                       X_shape=attr["X_shape"],
                                                       Y_shape=attr["Y_shape"],
                                                       batch_size=attr["batch_size"],
                                                       mul_type=attr["mul_type"],
                                                       is_constant=attr["is_constant"],
                                                       batch_axis=attr["batch_axis"])
        print("num_batch", num_batch)
    global_iters = num_batch * num_epoch
    return mul_ops, global_iters, num_batch


def fill_op_beaver_triple_matrix_shape(mul_ops: dict, *, X_shape, Y_shape, batch_size, op_id, mul_type,
                                       is_constant=False, batch_axis=0):
    assert len(X_shape) == len(Y_shape)
    num_dim = len(X_shape)

    mul_ops[op_id] = dict()
    if is_constant:
        mul_ops[op_id]["is_constant"] = is_constant
        mul_ops[op_id]["num_dim"] = num_dim
        mul_ops[op_id]["mul_type"] = mul_type
        mul_ops[op_id]["left_0"] = X_shape[0]
        mul_ops[op_id]["left_1"] = X_shape[1]
        mul_ops[op_id]["right_0"] = Y_shape[0]
        mul_ops[op_id]["right_1"] = Y_shape[1]
        mul_ops[op_id]["last_left_0"] = X_shape[0]
        mul_ops[op_id]["last_left_1"] = X_shape[1]
        mul_ops[op_id]["last_right_0"] = Y_shape[0]
        mul_ops[op_id]["last_right_1"] = Y_shape[1]

        if num_dim == 3:
            mul_ops[op_id]["left_2"] = X_shape[2]
            mul_ops[op_id]["last_left_2"] = X_shape[2]
            mul_ops[op_id]["right_2"] = Y_shape[2]
            mul_ops[op_id]["last_right_2"] = Y_shape[2]

        # return number of batch, which is 1
        return 1

    residual = X_shape[0] % batch_size
    if residual == 0:
        # if residual is 0, the number of samples is in multiples of batch_size.
        # Thus, we can directly use the real floor division operator "//" to compute
        # the num_batch
        num_batch = X_shape[0] // batch_size

        # if residual is 0, the last batch is a full batch.
        # In other words, all batches have the same number of samples
        last_batch_size = batch_size
    else:
        # if residual is not 0,
        num_batch = X_shape[0] // batch_size + 1

        # if residual is not 0, the last batch has 'residual' number of samples
        last_batch_size = residual

    print("num_batch", num_batch)

    mul_ops[op_id] = dict()
    if mul_type == "matmul":
        if num_dim == 2:
            if batch_axis == 0:
                assert X_shape[1] == Y_shape[0]
                mul_ops[op_id]["num_dim"] = 2
                mul_ops[op_id]["mul_type"] = "matmul"
                mul_ops[op_id]["left_0"] = batch_size
                mul_ops[op_id]["left_1"] = X_shape[1]
                mul_ops[op_id]["right_0"] = Y_shape[0]
                mul_ops[op_id]["right_1"] = Y_shape[1]
                mul_ops[op_id]["last_left_0"] = last_batch_size
                mul_ops[op_id]["last_left_1"] = X_shape[1]
                mul_ops[op_id]["last_right_0"] = Y_shape[0]
                mul_ops[op_id]["last_right_1"] = Y_shape[1]
            elif batch_axis == 1:
                mul_ops[op_id]["num_dim"] = 2
                mul_ops[op_id]["mul_type"] = "matmul"
                mul_ops[op_id]["left_0"] = X_shape[0]
                mul_ops[op_id]["left_1"] = batch_size
                mul_ops[op_id]["right_0"] = batch_size
                mul_ops[op_id]["right_1"] = Y_shape[1]
                mul_ops[op_id]["last_left_0"] = X_shape[0]
                mul_ops[op_id]["last_left_1"] = last_batch_size
                mul_ops[op_id]["last_right_0"] = last_batch_size
                mul_ops[op_id]["last_right_1"] = Y_shape[1]
            else:
                raise TypeError(
                    "currently does not support batch_axis {0} for {1} operation with {2} number of dimensions".format(
                        batch_axis, mul_type, num_dim))

        elif num_dim == 3:
            if batch_axis == 0:
                assert X_shape[0] == Y_shape[0]
                assert X_shape[2] == Y_shape[1]
                mul_ops[op_id]["num_dim"] = 3
                mul_ops[op_id]["mul_type"] = "matmul"
                mul_ops[op_id]["left_0"] = batch_size
                mul_ops[op_id]["left_1"] = X_shape[1]
                mul_ops[op_id]["left_2"] = X_shape[2]
                mul_ops[op_id]["right_0"] = batch_size
                mul_ops[op_id]["right_1"] = Y_shape[1]
                mul_ops[op_id]["right_2"] = Y_shape[2]
                mul_ops[op_id]["last_left_0"] = last_batch_size
                mul_ops[op_id]["last_left_1"] = X_shape[1]
                mul_ops[op_id]["last_left_2"] = X_shape[2]
                mul_ops[op_id]["last_right_0"] = last_batch_size
                mul_ops[op_id]["last_right_1"] = Y_shape[1]
                mul_ops[op_id]["last_right_2"] = Y_shape[2]
            else:
                raise TypeError(
                    "currently does not support batch_axis {0} for {1} operation with {2} number of dimensions".format(
                        batch_axis, mul_type, num_dim))
        else:
            raise TypeError("currently does not support num_dim {0} for mul_type {1}".format(num_dim, mul_type))

    elif mul_type == "multiply":
        assert X_shape == Y_shape
        if num_dim == 2:
            if batch_axis == 0:
                mul_ops[op_id]["num_dim"] = 2
                mul_ops[op_id]["mul_type"] = "multiply"
                mul_ops[op_id]["left_0"] = batch_size
                mul_ops[op_id]["left_1"] = X_shape[1]
                mul_ops[op_id]["right_0"] = batch_size
                mul_ops[op_id]["right_1"] = X_shape[1]
                mul_ops[op_id]["last_left_0"] = last_batch_size
                mul_ops[op_id]["last_left_1"] = Y_shape[1]
                mul_ops[op_id]["last_right_0"] = last_batch_size
                mul_ops[op_id]["last_right_1"] = Y_shape[1]
            else:
                raise TypeError(
                    "currently does not support batch_axis {0} for {1} operation with {2} number of dimensions".format(
                        batch_axis, mul_type, num_dim))

        elif num_dim == 3:
            if batch_axis == 0:
                mul_ops[op_id]["num_dim"] = 3
                mul_ops[op_id]["mul_type"] = "multiply"
                mul_ops[op_id]["left_0"] = batch_size
                mul_ops[op_id]["left_1"] = X_shape[1]
                mul_ops[op_id]["left_2"] = X_shape[2]
                mul_ops[op_id]["right_0"] = batch_size
                mul_ops[op_id]["right_1"] = Y_shape[1]
                mul_ops[op_id]["right_2"] = Y_shape[2]
                mul_ops[op_id]["last_left_0"] = last_batch_size
                mul_ops[op_id]["last_left_1"] = X_shape[1]
                mul_ops[op_id]["last_left_2"] = X_shape[2]
                mul_ops[op_id]["last_right_0"] = last_batch_size
                mul_ops[op_id]["last_right_1"] = Y_shape[1]
                mul_ops[op_id]["last_right_2"] = Y_shape[2]
            else:
                raise TypeError(
                    "currently does not support batch_axis {0} for {1} operation with {2} number of dimensions".format(
                        batch_axis, mul_type, num_dim))
        else:
            raise TypeError("currently does not support num_dim {0} for mul_type {1}".format(num_dim, mul_type))
    else:
        raise TypeError("currently does not support ", mul_type)

    return num_batch


def generate_shares(num_dim, ret):
    if num_dim == 2:
        k, l, p, q = ret
        print("k, p, l, q", k, l, p, q)

        X = generate_random_matrix(k, l)
        X0, X1 = share(X)

        Y = generate_random_matrix(p, q)
        Y0, Y1 = share(Y)

    elif num_dim == 3:
        j, k, l, o, p, q = ret
        print("j, k, l, o, p, q:", j, k, l, o, p, q)

        X = generate_random_3_dim_matrix(j, k, l)
        X0, X1 = share(X)

        Y = generate_random_3_dim_matrix(o, p, q)
        Y0, Y1 = share(Y)

    else:
        raise TypeError()

    return X, X0, X1, Y, Y0, Y1


class BaseBeaverTripleGenerationHelper(object):

    def __init__(self, mul_ops, global_iters, num_batch):
        self.global_iters = global_iters
        self.mul_ops = mul_ops
        self.num_batch = num_batch

    def initialize_beaver_triples(self):
        for i in range(self.global_iters):
            for op_id, val in self.mul_ops.items():
                ret = get_matrix_shapes(val=val, global_iter_index=i, num_batch=self.num_batch)
                X, X0, X1, Y, Y0, Y1 = generate_shares(val["num_dim"], ret)
                self._initialize_beaver_triples(i, op_id, X, X0, X1, Y, Y0, Y1)

    def complete_beaver_triples(self, other_party_transfer_bt_map, carlo_transfer_bt_map):
        for i in range(self.global_iters):
            for op_id in self.mul_ops.keys():
                self._complete_beaver_triples(i, op_id, other_party_transfer_bt_map, carlo_transfer_bt_map)

    def _initialize_beaver_triples(self, index, op_id, X, X0, X1, Y, Y0, Y1):
        pass

    def _complete_beaver_triples(self, index, op_id, other_party_transfer_bt_map, carlo_transfer_bt_map):
        pass


class PartyABeaverTripleGenerationHelper(BaseBeaverTripleGenerationHelper):

    def __init__(self, mul_ops, global_iters, num_batch):
        super(PartyABeaverTripleGenerationHelper, self).__init__(mul_ops, global_iters, num_batch)
        self.party_a_bt_map = [dict() for _ in range(global_iters)]
        self.party_a_bt_map_to_carlo = [dict() for _ in range(global_iters)]
        self.party_a_bt_map_to_b = [dict() for _ in range(global_iters)]

    def _initialize_beaver_triples(self, index, op_id, X, X0, X1, Y, Y0, Y1):

        self.party_a_bt_map[index][op_id] = dict()
        self.party_a_bt_map[index][op_id]["A0"] = X
        self.party_a_bt_map[index][op_id]["B0"] = Y
        self.party_a_bt_map[index][op_id]["U0"] = X0
        self.party_a_bt_map[index][op_id]["U1"] = X1
        self.party_a_bt_map[index][op_id]["E0"] = Y0
        self.party_a_bt_map[index][op_id]["E1"] = Y1

        # exchange
        self.party_a_bt_map_to_b[index][op_id] = dict()
        self.party_a_bt_map_to_b[index][op_id]["U1"] = X1
        self.party_a_bt_map_to_b[index][op_id]["E1"] = Y1

        # to carlo
        self.party_a_bt_map_to_carlo[index][op_id] = dict()
        self.party_a_bt_map_to_carlo[index][op_id]["U0"] = X0
        self.party_a_bt_map_to_carlo[index][op_id]["E0"] = Y0

    def _complete_beaver_triples(self, index, op_id, party_b_bt_map_to_a, carlo_transfer_bt_map):
        # P0 compute W0, Z0 and C0
        A0 = self.party_a_bt_map[index][op_id]["A0"]
        B0 = self.party_a_bt_map[index][op_id]["B0"]

        U0 = self.party_a_bt_map[index][op_id]["U0"]
        U1 = self.party_a_bt_map[index][op_id]["U1"]
        V0 = party_b_bt_map_to_a[index][op_id]["V0"]
        S0 = carlo_transfer_bt_map[index][op_id]["S0"]

        E0 = self.party_a_bt_map[index][op_id]["E0"]
        E1 = self.party_a_bt_map[index][op_id]["E1"]
        F0 = party_b_bt_map_to_a[index][op_id]["F0"]
        K0 = carlo_transfer_bt_map[index][op_id]["K0"]

        mul_type = self.mul_ops[op_id]["mul_type"]
        if mul_type == "matmul":
            W0 = np.matmul(U0, V0) + np.matmul(U1, V0) + S0
            Z0 = np.matmul(F0, E0) + np.matmul(F0, E1) + K0
            C0 = np.matmul(A0, B0) + W0 + Z0
        elif mul_type == "multiply":
            W0 = np.multiply(U0, V0) + np.multiply(U1, V0) + S0
            Z0 = np.multiply(F0, E0) + np.multiply(F0, E1) + K0
            C0 = np.multiply(A0, B0) + W0 + Z0
        else:
            raise TypeError("does not support" + mul_type)

        self.party_a_bt_map[index][op_id]["C0"] = C0

    def get_beaver_triple_map(self):
        return self.party_a_bt_map

    def get_to_carlo_beaver_triple_map(self):
        return self.party_a_bt_map_to_carlo

    def get_to_other_party_beaver_triple_map(self):
        return self.party_a_bt_map_to_b


class PartyBBeaverTripleGenerationHelper(BaseBeaverTripleGenerationHelper):

    def __init__(self, mul_ops, global_iters, num_batch):
        super(PartyBBeaverTripleGenerationHelper, self).__init__(mul_ops, global_iters, num_batch)
        self.party_b_bt_map = [dict() for _ in range(global_iters)]
        self.party_b_bt_map_to_carlo = [dict() for _ in range(global_iters)]
        self.party_b_bt_map_to_a = [dict() for _ in range(global_iters)]

    def _initialize_beaver_triples(self, index, op_id, X, X0, X1, Y, Y0, Y1):
        self.party_b_bt_map[index][op_id] = dict()
        self.party_b_bt_map[index][op_id]["A1"] = X
        self.party_b_bt_map[index][op_id]["B1"] = Y
        self.party_b_bt_map[index][op_id]["V0"] = Y0
        self.party_b_bt_map[index][op_id]["V1"] = Y1
        self.party_b_bt_map[index][op_id]["F0"] = X0
        self.party_b_bt_map[index][op_id]["F1"] = X1

        # exchange
        self.party_b_bt_map_to_a[index][op_id] = dict()
        self.party_b_bt_map_to_a[index][op_id]["F0"] = X0
        self.party_b_bt_map_to_a[index][op_id]["V0"] = Y0

        # to carlo
        self.party_b_bt_map_to_carlo[index][op_id] = dict()
        self.party_b_bt_map_to_carlo[index][op_id]["V1"] = Y1
        self.party_b_bt_map_to_carlo[index][op_id]["F1"] = X1

    def _complete_beaver_triples(self, index, op_id, party_a_bt_map_to_b, carlo_transfer_bt_map):
        # P1 compute W1, Z1 and C1

        A1 = self.party_b_bt_map[index][op_id]["A1"]
        B1 = self.party_b_bt_map[index][op_id]["B1"]

        U1 = party_a_bt_map_to_b[index][op_id]["U1"]
        V1 = self.party_b_bt_map[index][op_id]["V1"]
        S1 = carlo_transfer_bt_map[index][op_id]["S1"]

        E1 = party_a_bt_map_to_b[index][op_id]["E1"]
        F1 = self.party_b_bt_map[index][op_id]["F1"]
        K1 = carlo_transfer_bt_map[index][op_id]["K1"]

        mul_type = self.mul_ops[op_id]["mul_type"]
        if mul_type == "matmul":
            W1 = np.matmul(U1, V1) + S1
            Z1 = np.matmul(F1, E1) + K1
            C1 = np.matmul(A1, B1) + W1 + Z1
        elif mul_type == "multiply":
            W1 = np.multiply(U1, V1) + S1
            Z1 = np.multiply(F1, E1) + K1
            C1 = np.multiply(A1, B1) + W1 + Z1
        else:
            raise TypeError("does not support" + mul_type)

        self.party_b_bt_map[index][op_id]["C1"] = C1

    def get_beaver_triple_map(self):
        return self.party_b_bt_map

    def get_to_carlo_beaver_triple_map(self):
        return self.party_b_bt_map_to_carlo

    def get_to_other_party_beaver_triple_map(self):
        return self.party_b_bt_map_to_a


def create_beaver_triples(mul_ops, global_iters, num_batch):
    """
    :param mul_ops:
    :param global_iters:  start from 0
    :param num_batch
    :return:
    """
    print("global_iters, num_batch", global_iters, num_batch)
    party_a_bt_gene_helper = PartyABeaverTripleGenerationHelper(mul_ops, global_iters, num_batch)
    party_a_bt_gene_helper.initialize_beaver_triples()

    party_b_bt_gene_helper = PartyBBeaverTripleGenerationHelper(mul_ops, global_iters, num_batch)
    party_b_bt_gene_helper.initialize_beaver_triples()

    party_a_bt_map_to_carlo = party_a_bt_gene_helper.get_to_carlo_beaver_triple_map()
    party_a_bt_map_to_b = party_a_bt_gene_helper.get_to_other_party_beaver_triple_map()

    party_b_bt_map_to_carlo = party_b_bt_gene_helper.get_to_carlo_beaver_triple_map()
    party_b_bt_map_to_a = party_b_bt_gene_helper.get_to_other_party_beaver_triple_map()

    carlo_bt_map_to_party_a, carlo_bt_map_to_party_b = carlo_deal_data(party_a_bt_map_to_carlo,
                                                                       party_b_bt_map_to_carlo, mul_ops)

    party_a_bt_gene_helper.complete_beaver_triples(party_b_bt_map_to_a, carlo_bt_map_to_party_a)
    party_a_bt_map = party_a_bt_gene_helper.get_beaver_triple_map()

    party_b_bt_gene_helper.complete_beaver_triples(party_a_bt_map_to_b, carlo_bt_map_to_party_b)
    party_b_bt_map = party_b_bt_gene_helper.get_beaver_triple_map()
    return party_a_bt_map, party_b_bt_map
