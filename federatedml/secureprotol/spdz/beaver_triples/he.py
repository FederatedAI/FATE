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

from federatedml.secureprotol.spdz.communicator import Communicator
from federatedml.secureprotol.spdz.utils.random_device import rand_tensor


def beaver_triplets(a_shape, b_shape, einsum_expr, q_field, he_key_pair, communicator: Communicator, name):
    """
    a = a_1 + a_2 + ... + a_n
    b = b_1 + b_2 + ... + b_n
    c = c_1 + c_2 + ... + c_n
    subject to
        c_i = a_i * b_i
            + sum([Dec(a_j * Enc(b_i) + r_{ij}) for j in range(n) if i != j])
            - sum([r_{ji} for j in range(n) if j != i])
        one has
        c = a * b
    """
    public_key, private_key = he_key_pair
    a = rand_tensor(q_field, a_shape).astype(object)
    b = rand_tensor(q_field, b_shape).astype(object)

    def _einsum(_a, _b):
        return np.einsum(einsum_expr, _a, _b, optimize="optimize")

    c = _einsum(a, b)

    # tensor dot of local share of a with encrypted remote share of b
    def _cross_terms(_b):
        return _einsum(a, _b)

    # broadcast encrypted b
    encrypted_b = np.vectorize(public_key.encrypt)(b)
    communicator.remote_encrypted_tensor(encrypted=encrypted_b, tag=name)

    # get encrypted b
    parties, encrypted_b_list = communicator.get_encrypted_tensors(tag=name)
    for _b, _p in zip(encrypted_b_list, parties):
        cross = _cross_terms(_b)
        r = rand_tensor(q_field, cross.shape).astype(object)
        cross += r
        c -= r
        # remote cross terms
        communicator.remote_encrypted_cross_tensor(encrypted=cross, parties=_p, tag=name)

    # get cross terms
    crosses = communicator.get_encrypted_cross_tensors(tag=name)
    for cross in crosses:
        c += np.vectorize(private_key.decrypt, otypes=[object])(cross)

    return a, b, c % q_field
