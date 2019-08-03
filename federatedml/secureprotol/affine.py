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

import random
import math

from federatedml.secureprotol.affine_encoder import AffineEncoder
from federatedml.secureprotol import gmpy_math


class AffineCipher(object):
    def __init__(self):
        pass

    @staticmethod
    def generate_keypair(key_size=2048, encode_precision=2 ** 100, a_ratio=None, b_ratio=None):
        n = random.SystemRandom().getrandbits(key_size)
        if a_ratio is None:
            a_ratio = random.SystemRandom().random()
        if b_ratio is None:
            b_ratio = random.SystemRandom().random()
        while True:
            a = random.SystemRandom().getrandbits(int(key_size * a_ratio))
            if math.gcd(n, a) == 1:
                break
        b = random.SystemRandom().getrandbits(int(key_size * b_ratio))
        return AffineCipherKey(a, b, n, encode_precision)


class AffineCipherKey(object):
    def __init__(self, a, b, n, encode_precision=2 ** 100):
        self.a = a
        self.b = b
        self.n = n
        self.a_inv = self.mod_inverse()
        self.affine_encoder = AffineEncoder(mult=encode_precision)

    def encrypt(self, plaintext):
        return self.raw_encrypt(self.affine_encoder.encode(plaintext))

    def decrypt(self, ciphertext):
        return self.affine_encoder.decode(self.raw_decrypt(ciphertext), ciphertext.multiplier)

    def raw_encrypt(self, plaintext):
        return AffineCiphertext((self.a * plaintext + self.b) % self.n)

    def raw_decrypt(self, ciphertext):
        plaintext = (self.a_inv * (ciphertext.cipher % self.n - ciphertext.multiplier * self.b)) % self.n
        if plaintext / self.n > 0.9:
            return plaintext - self.n
        return plaintext

    def mod_inverse(self):
        return gmpy_math.invert(self.a, self.n)


class AffineCiphertext(object):
    def __init__(self, cipher, multiplier=1):
        self.cipher = cipher
        self.multiplier = multiplier

    def __add__(self, other):
        if isinstance(other, AffineCiphertext):
            return AffineCiphertext(self.cipher + other.cipher, self.multiplier + other.multiplier)
        elif type(other) is int and other == 0:
            return self
        else:
            raise TypeError("Addition only supports AffineCiphertext and initialization with int zero")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if type(other) is int:
            return AffineCiphertext(self.cipher * other, self.multiplier * other)
        else:
            raise TypeError("Multiplication only supports int.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __truediv__(self, other):
        return self.__mul__(1 / other)
    
    