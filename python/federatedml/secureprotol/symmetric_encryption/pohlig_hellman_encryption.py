#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from federatedml.secureprotol.gmpy_math import is_prime, invert, gcd, powmod, next_prime
from federatedml.secureprotol.symmetric_encryption.symmetric_encryption import SymmetricKey, SymmetricCiphertext
from federatedml.util import conversion


class PohligHellmanCipherKey(SymmetricKey):
    """
    A commutative encryption scheme inspired by Pohlig, Stephen, and Martin Hellman. "An improved algorithm
        for computing logarithms over GF (p) and its cryptographic significance." 1978
    Enc(x) = x^a mod p, with public knowledge p being a prime and satisfying that (p - 1) / 2 is also a prime
    Dec(y) = y^(a^(-1) mod phi(p)) mod p
    """

    def __init__(self, mod_base, exponent=None):
        """

        :param exponent: int
        :param mod_base: int
        """
        super(PohligHellmanCipherKey, self).__init__()
        self.mod_base = mod_base    # p
        if exponent is not None and gcd(exponent, mod_base - 1) != 1:
            raise ValueError("In Pohlig, exponent and the totient of the modulo base must be coprimes")
        self.exponent = exponent    # a
        self.exponent_inverse = None if exponent is None else invert(exponent, mod_base - 1)

    @staticmethod
    def generate_key(key_size=1024):
        """
        Generate a self-typed object with public mod_base and vacant exponent
        :param key_size: int
        :return: PohligHellmanCipherKey
        """
        key_size_half = key_size // 2
        while True:
            mod_base_half = PohligHellmanCipherKey.generate_prime(2 ** (key_size_half - 1), 2 ** key_size_half - 1)
            mod_base = mod_base_half * 2 + 1
            if is_prime(mod_base):
                return PohligHellmanCipherKey(mod_base)

    @staticmethod
    def generate_prime(left, right):
        """
        Generate a prime over (left, right]
        :param left:
        :param right:
        :return:
        """
        while True:
            random_integer = random.randint(left, right)
            random_prime = next_prime(random_integer)
            if random_prime <= right:
                return random_prime

    def init(self):
        """
        Init self.exponent
        :return:
        """
        while True:
            self.exponent = random.randint(2, self.mod_base)
            if gcd(self.exponent, self.mod_base - 1) == 1:
                self.exponent_inverse = invert(self.exponent, self.mod_base - 1)
                break

    def encrypt(self, plaintext):
        if isinstance(plaintext, list):
            return self.encrypt_list(plaintext)
        return self.encrypt_single_val(plaintext)

    def encrypt_single_val(self, plaintext):
        """

        :param plaintext: int >= 0 / str / PohligHellmanCiphertext
        :return: PohligHellmanCiphertext
        """
        if isinstance(plaintext, str):
            plaintext = conversion.str_to_int(plaintext)
        elif isinstance(plaintext, PohligHellmanCiphertext):
            plaintext = plaintext.message
        elif not isinstance(plaintext, int):
            plaintext = conversion.str_to_int(str(plaintext))

        ciphertext = powmod(plaintext, self.exponent, self.mod_base)
        return PohligHellmanCiphertext(ciphertext)

    def encrypt_list(self, list_plaintext):
        ciphertext = [self.encrypt_single_val(p) for p in list_plaintext]
        return ciphertext

    def decrypt(self, ciphertext, decode_output=False):
        if isinstance(ciphertext, list):
            return self.decrypt_list(ciphertext, decode_output)
        return self.decrypt_single_val(ciphertext, decode_output)

    def decrypt_single_val(self, ciphertext, decode_output=False):
        """
        If decode, then call int_to_str() method to decode the output plaintext
        :param ciphertext: PohligHellmanCiphertext
        :param decode_output: bool
        :return: PohligHellmanCiphertext / str
        """
        if isinstance(ciphertext, PohligHellmanCiphertext):
            ciphertext = ciphertext.message
        elif isinstance(ciphertext, str):
            ciphertext = conversion.str_to_int(ciphertext)

        if decode_output:
            return conversion.int_to_str(powmod(ciphertext, self.exponent_inverse, self.mod_base))
        else:
            return PohligHellmanCiphertext(powmod(ciphertext, self.exponent_inverse, self.mod_base))

    def decrypt_list(self, ciphertext, decode_output):
        decrypt_result = [self.decrypt_single_val(c, decode_output) for c in ciphertext]
        return decrypt_result


class PohligHellmanCiphertext(SymmetricCiphertext):
    """

    """

    def __init__(self, message):
        super(PohligHellmanCiphertext, self).__init__()
        self.message = message

    def __hash__(self):
        return self.message.__hash__()

    def __eq__(self, other):
        if not isinstance(other, PohligHellmanCiphertext):
            raise TypeError("Can only compare two PohligHellmanCiphertext objects")

        if self.message == other.message:
            return True
        else:
            return False
