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
import math
import random
import numpy as np

from federatedml.secureprotol.affine_encoder import AffineEncoder
from federatedml.secureprotol.gmpy_math import invert


class IterativeAffineCipher(object):
    """
    Formulas: The r-th round of encryption method is Enc_r(x) = a_r * x % n_r;
            The overall encryption scheme is Enc(x) = Enc_n o ... o Enc_1(x)
    Note: The key round supported is upper bounded by some number dependent of key size.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_keypair(key_size=1024, key_round=5, encode_precision=2 ** 100):
        key_size_array = np.linspace(start=int(key_size / 2), stop=key_size, num=key_round)
        key_size_array = np.floor(key_size_array).astype(np.int64)
        n_array = [0 for _ in range(key_round)]
        a_array = [0 for _ in range(key_round)]
        i = 0
        for key_size in key_size_array:
            n = random.SystemRandom().getrandbits(key_size)
            a_ratio = random.SystemRandom().random()
            a = 0
            while True:
                a_size = int(key_size * a_ratio)
                if a_size is 0:
                    continue
                a = random.SystemRandom().getrandbits(a_size)
                if math.gcd(n, a) == 1:
                    break
            n_array[i] = n
            a_array[i] = a
            i = i + 1
        return IterativeAffineCipherKey(a_array, n_array, encode_precision)


class IterativeAffineCipherKey(object):
    def __init__(self, a_array, n_array, encode_precision=2 ** 100):
        if len(a_array) != len(n_array):
            raise ValueError("a_array length must be equal to n_array")
        self.a_array = a_array
        self.n_array = n_array
        self.key_round = len(self.a_array)
        self.a_inv_array = self.mod_inverse()
        self.affine_encoder = AffineEncoder(mult=encode_precision)

    def encrypt(self, plaintext):
        return self.raw_encrypt(self.affine_encoder.encode(plaintext))

    def decrypt(self, ciphertext):
        if isinstance(ciphertext, int) is True and ciphertext is 0:
            return 0
        return self.affine_encoder.decode(self.raw_decrypt(ciphertext))

    def raw_encrypt(self, plaintext):
        ciphertext = IterativeAffineCiphertext(plaintext)
        for i in range(self.key_round):
            ciphertext = self.raw_encrypt_round(ciphertext, i)
        return ciphertext

    def raw_decrypt(self, ciphertext):
        plaintext = ciphertext.cipher
        for i in range(self.key_round):
            plaintext = self.raw_decrypt_round(plaintext, i)
        if plaintext / self.n_array[0] > 0.9:
            return plaintext - self.n_array[0]
        else:
            return plaintext

    def raw_encrypt_round(self, plaintext, round_index):
        return IterativeAffineCiphertext((self.a_array[round_index] * plaintext.cipher) % self.n_array[round_index])

    def raw_decrypt_round(self, ciphertext, round_index):
        plaintext = (self.a_inv_array[self.key_round - 1 - round_index]
                     * (ciphertext % self.n_array[self.key_round - 1 - round_index]))\
                    % self.n_array[self.key_round - 1 - round_index]
        return plaintext

    def mod_inverse(self):
        a_array_inv = [0 for _ in self.a_array]
        for i in range(self.key_round):
            a_array_inv[i] = invert(self.a_array[i], self.n_array[i])
        return a_array_inv


class IterativeAffineCiphertext(object):
    def __init__(self, cipher):
        self.cipher = cipher

    def __add__(self, other):
        if isinstance(other, IterativeAffineCiphertext):
            return IterativeAffineCiphertext(self.cipher + other.cipher)
        elif type(other) is int and other == 0:
            return self
        else:
            raise TypeError("Addition only supports AffineCiphertext and initialization with int zero")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __mul__(self, other):
        if type(other) is int and other is -1:
            return IterativeAffineCiphertext(self.cipher * other)
        else:
            raise TypeError("Multiplication only supports int -1.")

    def __rmul__(self, other):
        return self.__mul__(other)
