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

from arch.api.table.table import Table
from federatedml.secureprotol.spdz.communicator import Communicator
from federatedml.secureprotol.spdz.utils.random_utils import rand_tensor


def encrypt_tensor(tensor, public_key):
    if isinstance(tensor, np.ndarray):
        return np.vectorize(public_key.encrypt)(tensor)
    elif isinstance(tensor, Table):
        return tensor.mapValues(lambda x: np.vectorize(public_key.encrypt)(x))
    else:
        raise NotImplementedError(f"type={type(tensor)}")


def decrypt_tensor(tensor, private_key, otypes):
    if isinstance(tensor, np.ndarray):
        return np.vectorize(private_key.decrypt, otypes)(tensor)
    elif isinstance(tensor, Table):
        return tensor.mapValues(lambda x: np.vectorize(private_key.decrypt, otypes)(x))
    else:
        raise NotImplementedError(f"type={type(tensor)}")


def beaver_triplets(a_tensor, b_tensor, dot, q_field, he_key_pair, communicator: Communicator, name):
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
    a = rand_tensor(q_field, a_tensor)
    b = rand_tensor(q_field, b_tensor)

    c = dot(a, b)

    # tensor dot of local share of a with encrypted remote share of b
    def _cross_terms(_b):
        return dot(a, _b)

    # broadcast encrypted b
    encrypted_b = encrypt_tensor(b, public_key)
    communicator.remote_encrypted_tensor(encrypted=encrypted_b, tag=name)

    # get encrypted b
    parties, encrypted_b_list = communicator.get_encrypted_tensors(tag=name)
    for _b, _p in zip(encrypted_b_list, parties):
        cross = _cross_terms(_b)
        r = rand_tensor(q_field, cross)
        cross += r
        c -= r
        # remote cross terms
        communicator.remote_encrypted_cross_tensor(encrypted=cross, parties=_p, tag=name)

    # get cross terms
    crosses = communicator.get_encrypted_cross_tensors(tag=name)
    for cross in crosses:
        c += decrypt_tensor(cross, private_key, [object])

    return a, b, c % q_field
