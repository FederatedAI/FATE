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
import torch
from collections import Iterable
from Cryptodome import Random
from Cryptodome.PublicKey import RSA

# from arch.api.utils import log_utils
from federatedml.secureprotol import gmpy_math
from federatedml.secureprotol.affine import AffineCipher
from federatedml.secureprotol.fate_paillier import PaillierKeypair
from federatedml.secureprotol.random import RandomPads


# LOGGER = log_utils.getLogger()
from federatedml.secureprotol.iterative_affine import IterativeAffineCipher


class Encrypt(object):
    def __init__(self):
        self.public_key = None
        self.privacy_key = None

    def generate_key(self, n_length=0):
        pass

    def set_public_key(self, public_key):
        pass

    def get_public_key(self):
        pass

    def set_privacy_key(self, privacy_key):
        pass

    def get_privacy_key(self):
        pass

    def encrypt(self, value):
        pass

    def decrypt(self, value):
        pass

    def encrypt_list(self, values):
        result = [self.encrypt(msg) for msg in values]
        return result

    def decrypt_list(self, values):
        result = [self.decrypt(msg) for msg in values]
        return result

    def distribute_decrypt(self, X):
        decrypt_table = X.mapValues(lambda x: self.decrypt(x))
        return decrypt_table

    def distribute_encrypt(self, X):
        encrypt_table = X.mapValues(lambda x: self.encrypt(x))
        return encrypt_table

    def _recursive_func(self, obj, func):
        if isinstance(obj, np.ndarray):
            if len(obj.shape) == 1:
                return np.reshape([func(val) for val in obj], obj.shape)
            else:
                return np.reshape([self._recursive_func(o, func) for o in obj], obj.shape)
        elif isinstance(obj, Iterable):
            return type(obj)(
                self._recursive_func(o, func) if isinstance(o, Iterable) else func(o) for o in obj)
        else:
            return func(obj)

    def recursive_encrypt(self, X):
        return self._recursive_func(X, self.encrypt)

    def recursive_decrypt(self, X):
        return self._recursive_func(X, self.decrypt)


class RsaEncrypt(Encrypt):
    def __init__(self):
        super(RsaEncrypt, self).__init__()
        self.e = None
        self.d = None
        self.n = None

    def generate_key(self, rsa_bit=1024):
        random_generator = Random.new().read
        rsa = RSA.generate(rsa_bit, random_generator)
        self.e = rsa.e
        self.d = rsa.d
        self.n = rsa.n

    def get_key_pair(self):
        return self.e, self.d, self.n

    def set_public_key(self, public_key):
        self.e = public_key["e"]
        self.n = public_key["n"]

    def get_public_key(self):
        return self.e, self.n

    def set_privacy_key(self, privacy_key):
        self.d = privacy_key["d"]
        self.n = privacy_key["n"]

    def get_privacy_key(self):
        return self.d, self.n

    def encrypt(self, value):
        if self.e is not None and self.n is not None:
            return gmpy_math.powmod(value, self.e, self.n)
        else:
            return None

    def decrypt(self, value):
        if self.d is not None and self.n is not None:
            return gmpy_math.powmod(value, self.d, self.n)
        else:
            return None


class PaillierEncrypt(Encrypt):
    def __init__(self):
        super(PaillierEncrypt, self).__init__()

    def generate_key(self, n_length=1024):
        self.public_key, self.privacy_key = \
            PaillierKeypair.generate_keypair(n_length=n_length)

    def get_key_pair(self):
        return self.public_key, self.privacy_key

    def set_public_key(self, public_key):
        self.public_key = public_key

    def get_public_key(self):
        return self.public_key

    def set_privacy_key(self, privacy_key):
        self.privacy_key = privacy_key

    def get_privacy_key(self):
        return self.privacy_key

    def encrypt(self, value):
        if self.public_key is not None:
            return self.public_key.encrypt(value)
        else:
            return None

    def decrypt(self, value):
        if self.privacy_key is not None:
            return self.privacy_key.decrypt(value)
        else:
            return None


class FakeEncrypt(Encrypt):
    def encrypt(self, value):
        return value

    def decrypt(self, value):
        return value


class SymmetricEncrypt(Encrypt):
    def __init__(self):
        self.key = None

    def encrypt(self, plaintext):
        pass


class AffineEncrypt(SymmetricEncrypt):
    def __init__(self):
        super(AffineEncrypt, self).__init__()

    def generate_key(self, key_size=1024):
        self.key = AffineCipher.generate_keypair(key_size=key_size)

    def encrypt(self, plaintext):
        if self.key is not None:
            return self.key.encrypt(plaintext)
        else:
            return None

    def decrypt(self, ciphertext):
        if self.key is not None:
            return self.key.decrypt(ciphertext)
        else:
            return None


class PadsCipher(Encrypt):

    def __init__(self):
        super().__init__()
        self._uuid = None
        self._rands = None

    def set_self_uuid(self, uuid):
        self._uuid = uuid

    def set_exchanged_keys(self, keys):
        self._seeds = {uid: v & 0xffffffff for uid, v in keys.items() if uid != self._uuid}
        self._rands = {uid: RandomPads(v & 0xffffffff) for uid, v in keys.items() if uid != self._uuid}

    def encrypt(self, value):
        if isinstance(value, np.ndarray):
            ret = value
            for uid, rand in self._rands.items():
                if uid > self._uuid:
                    ret = rand.add_rand_pads(ret, 1.0)
                else:
                    ret = rand.add_rand_pads(ret, -1.0)
            return ret
        elif isinstance(value, torch.Tensor):
            shape = value.shape
            value = value.view(-1)
            ret = value.numpy()
            for uid, rand in self._rands.items():
                if uid > self._uuid:
                    ret = rand.add_rand_pads(ret, 1.0)
                else:
                    ret = rand.add_rand_pads(ret, -1.0)
            ret = torch.Tensor(ret)
            return ret.reshape(shape)
        else:
            ret = value
            for uid, rand in self._rands.items():
                if uid > self._uuid:
                    ret += rand.rand(1)[0]
                else:
                    ret -= rand.rand(1)[0]
            return ret

    def decrypt(self, value):
        return value


class IterativeAffineEncrypt(SymmetricEncrypt):
    def __init__(self):
        super(IterativeAffineEncrypt, self).__init__()

    def generate_key(self, key_size=1024, key_round=5):
        self.key = IterativeAffineCipher.generate_keypair(key_size=key_size, key_round=key_round)

    def encrypt(self, plaintext):
        if self.key is not None:
            return self.key.encrypt(plaintext)
        else:
            return None

    def decrypt(self, ciphertext):
        if self.key is not None:
            return self.key.decrypt(ciphertext)
        else:
            return None
