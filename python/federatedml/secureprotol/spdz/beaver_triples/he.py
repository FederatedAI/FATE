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

from fate_arch.session import is_table
from federatedml.secureprotol.spdz.communicator import Communicator
from federatedml.secureprotol.spdz.utils import rand_tensor, urand_tensor
from federatedml.util import LOGGER


def encrypt_tensor(tensor, public_key):
    encrypted_zero = public_key.encrypt(0)
    if isinstance(tensor, np.ndarray):
        return np.vectorize(lambda e: encrypted_zero + e)(tensor)
    elif is_table(tensor):
        return tensor.mapValues(lambda x: np.vectorize(lambda e: encrypted_zero + e)(x))
    else:
        raise NotImplementedError(f"type={type(tensor)}")


def decrypt_tensor(tensor, private_key, otypes):
    if isinstance(tensor, np.ndarray):
        return np.vectorize(private_key.decrypt, otypes)(tensor)
    elif is_table(tensor):
        return tensor.mapValues(lambda x: np.vectorize(private_key.decrypt, otypes)(x))
    else:
        raise NotImplementedError(f"type={type(tensor)}")


def beaver_triplets(a_tensor, b_tensor, dot, q_field, he_key_pair, communicator: Communicator, name):
    public_key, private_key = he_key_pair
    a = rand_tensor(q_field, a_tensor)
    b = rand_tensor(q_field, b_tensor)

    def _cross(self_index, other_index):
        LOGGER.debug(f"_cross: a={a}, b={b}")
        _c = dot(a, b)
        encrypted_a = encrypt_tensor(a, public_key)
        communicator.remote_encrypted_tensor(encrypted=encrypted_a, tag=f"{name}_a_{self_index}")
        r = urand_tensor(q_field, _c)
        _p, (ea,) = communicator.get_encrypted_tensors(tag=f"{name}_a_{other_index}")
        eab = dot(ea, b)
        eab += r
        _c -= r
        communicator.remote_encrypted_cross_tensor(encrypted=eab,
                                                   parties=_p,
                                                   tag=f"{name}_cross_a_{other_index}_b_{self_index}")
        crosses = communicator.get_encrypted_cross_tensors(tag=f"{name}_cross_a_{self_index}_b_{other_index}")
        for eab in crosses:
            _c += decrypt_tensor(eab, private_key, [object])

        return _c

    c = _cross(communicator.party_idx, 1 - communicator.party_idx)

    return a, b, c % q_field
