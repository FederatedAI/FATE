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

from arch.api.base.table import Table
from federatedml.secureprotol.spdz.communicator import Communicator
from federatedml.secureprotol.spdz.utils.random_utils import rand_tensor, urand_tensor


def encrypt_tensor(tensor, public_key):
    encrypted_zero = public_key.encrypt(0)
    if isinstance(tensor, np.ndarray):
        return np.vectorize(lambda e: encrypted_zero + e)(tensor)
    elif isinstance(tensor, Table):
        return tensor.mapValues(lambda x: np.vectorize(lambda e: encrypted_zero + e)(x))
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
    public_key, private_key = he_key_pair
    a = rand_tensor(q_field, a_tensor)
    b = rand_tensor(q_field, b_tensor)

    c = dot(a, b)

    # broadcast encrypted a and encrypted b
    if communicator.party_idx == 0:
        encrypted_a = encrypt_tensor(a, public_key)
        encrypted_b = encrypt_tensor(b, public_key)
        communicator.remote_encrypted_tensor(encrypted=encrypted_a, tag=f"{name}_a")
        communicator.remote_encrypted_tensor(encrypted=encrypted_b, tag=f"{name}_b")

    # get encrypted a and b
    if communicator.party_idx == 1:
        r = urand_tensor(q_field, c)
        _p, encrypted_a_list = communicator.get_encrypted_tensors(tag=f"{name}_a")
        _, encrypted_b_list = communicator.get_encrypted_tensors(tag=f"{name}_b")
        cross = dot(encrypted_a_list[0], b) + dot(a, encrypted_b_list[0])
        cross += r
        c -= r
        communicator.remote_encrypted_cross_tensor(encrypted=cross, parties=_p, tag=name)

    if communicator.party_idx == 0:
        # get cross terms
        crosses = communicator.get_encrypted_cross_tensors(tag=name)
        for cross in crosses:
            c += decrypt_tensor(cross, private_key, [object])

    return a, b, c % q_field
