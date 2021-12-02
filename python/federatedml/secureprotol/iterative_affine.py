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
from federatedml.secureprotol.gmpy_math import invert, mpz


class IterativeAffineCipher(object):
    """
    Formulas: The r-th round of encryption method is Enc_r(x) = a_r * x % n_r;
            The overall encryption scheme is Enc(x) = Enc_n o ... o Enc_1(x)
    Note: The key round supported is upper bounded by some number dependent of key size.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_keypair(key_size=1024, key_round=5, encode_precision=2 ** 100, randomized=True):
        if randomized:
            return IterativeAffineCipher.generate_randomized_keypair(key_size, key_round, encode_precision)
        else:
            return IterativeAffineCipher.generate_deterministic_keypair(key_size, key_round, encode_precision)

    @staticmethod
    def generate_randomized_keypair(key_size, key_round, encode_precision):
        key_size_array = np.linspace(start=int(key_size / 2), stop=key_size, num=key_round)
        key_size_array = np.floor(key_size_array).astype(np.int64)
        n_array = [0 for _ in range(key_round)]
        a_array = [0 for _ in range(key_round)]
        i = 0
        for key_size in key_size_array:
            n = random.SystemRandom().getrandbits(key_size)
            while True:
                a_ratio = random.SystemRandom().random()
                a_size = int(key_size * a_ratio)
                if a_size is 0:
                    continue
                a = random.SystemRandom().getrandbits(a_size)
                if math.gcd(n, a) == 1:
                    break
            n_array[i] = n
            a_array[i] = a
            i = i + 1

        # pick a generator and a scalar
        g = random.SystemRandom().getrandbits(key_size // 10)
        x = random.SystemRandom().getrandbits(160)
        return RandomizedIterativeAffineCipherKey(a_array, n_array, g, x, encode_precision=encode_precision)

    @staticmethod
    def generate_deterministic_keypair(key_size, key_round, encode_precision):
        key_size_array = np.linspace(start=int(key_size / 2), stop=key_size, num=key_round)
        key_size_array = np.floor(key_size_array).astype(np.int64)
        n_array = [0 for _ in range(key_round)]
        a_array = [0 for _ in range(key_round)]
        i = 0
        for key_size in key_size_array:
            n = random.SystemRandom().getrandbits(key_size)
            while True:
                a_ratio = random.SystemRandom().random()
                a_size = int(key_size * a_ratio)
                if a_size is 0:
                    continue
                a = random.SystemRandom().getrandbits(a_size)
                if math.gcd(n, a) == 1:
                    break
            n_array[i] = n
            a_array[i] = a
            i = i + 1

        return DeterministicIterativeAffineCipherKey(a_array, n_array, encode_precision)


class IterativeAffineCipherKey(object):
    def __init__(self, a_array, n_array, encode_precision=2 ** 100):
        if len(a_array) != len(n_array):
            raise ValueError("a_array length must be equal to n_array")
        self.a_array = a_array
        self.n_array = n_array
        self.key_round = len(self.a_array)
        self.a_inv_array = self.mod_inverse()
        self.affine_encoder = AffineEncoder(mult=encode_precision)

    def mod_inverse(self):
        a_array_inv = [0 for _ in self.a_array]
        for i in range(self.key_round):
            a_array_inv[i] = invert(self.a_array[i], self.n_array[i])
        return a_array_inv

    def encrypt(self, plaintext):
        pass

    def decrypt(self, ciphertext):
        pass


class RandomizedIterativeAffineCipherKey(IterativeAffineCipherKey):
    def __init__(self, a_array, n_array, g, x, encode_precision=2 ** 100):
        super(RandomizedIterativeAffineCipherKey, self).__init__(a_array, n_array, encode_precision)
        self.g = g
        self.x = x
        self.h = g * x % self.n_array[0]

    def encrypt(self, plaintext):
        return self.raw_encrypt(self.affine_encoder.encode(plaintext))

    def decrypt(self, ciphertext):
        if isinstance(ciphertext, int) is True and ciphertext is 0:
            return 0

        return self.affine_encoder.decode(self.raw_decrypt(ciphertext), ciphertext.mult_times)

    def raw_encrypt(self, plaintext):
        plaintext = self.encode(plaintext)
        ciphertext = RandomizedIterativeAffineCiphertext(plaintext[0],
                                                         plaintext[1],
                                                         self.n_array[-1],
                                                         self.affine_encoder.mult)
        for i in range(self.key_round):
            ciphertext = self.raw_encrypt_round(ciphertext, i)
        return ciphertext

    def raw_decrypt(self, ciphertext):
        plaintext1 = ciphertext.cipher1
        plaintext2 = ciphertext.cipher2
        for i in range(self.key_round):
            plaintext1, plaintext2 = self.raw_decrypt_round(plaintext1, plaintext2, i)
        encoded_result = RandomizedIterativeAffineCiphertext(
            cipher1=plaintext1,
            cipher2=plaintext2,
            n_final=ciphertext.n_final,
            multiple=ciphertext.multiple,
            mult_times=ciphertext.mult_times
        )
        return self.decode(encoded_result)

    def encode(self, plaintext):
        y = random.SystemRandom().getrandbits(160)
        return int(mpz(y) * self.g % self.n_array[0]), (plaintext + y * self.h) % self.n_array[0]

    def decode(self, ciphertext):
        intermediate_result = (ciphertext.cipher2 - self.x * ciphertext.cipher1) % self.n_array[0]
        if intermediate_result / self.n_array[0] > 0.9:
            intermediate_result -= self.n_array[0]
        return intermediate_result / ciphertext.multiple ** ciphertext.mult_times

    def raw_encrypt_round(self, plaintext, round_index):
        return RandomizedIterativeAffineCiphertext(
            plaintext.cipher1,
            int(mpz(self.a_array[round_index]) * plaintext.cipher2 % self.n_array[round_index]),
            plaintext.n_final,
            self.affine_encoder.mult
        )

    def raw_decrypt_round(self, ciphertext1, ciphertext2, round_index):
        cur_n = self.n_array[self.key_round - 1 - round_index]
        cur_a_inv = self.a_inv_array[self.key_round - 1 - round_index]
        plaintext1 = ciphertext1 % cur_n
        plaintext2 = cur_a_inv * ciphertext2 % cur_n
        if plaintext1 / cur_n > 0.9:
            plaintext1 -= cur_n
        if plaintext2 / cur_n > 0.9:
            plaintext2 -= cur_n
        return plaintext1, plaintext2


class DeterministicIterativeAffineCipherKey(IterativeAffineCipherKey):
    def encrypt(self, plaintext):
        return self.raw_encrypt(self.affine_encoder.encode(plaintext))

    def decrypt(self, ciphertext):
        if isinstance(ciphertext, int) is True and ciphertext is 0:
            return 0
        return self.affine_encoder.decode(self.raw_decrypt(ciphertext), mult_times=ciphertext.mult_times)

    def raw_encrypt(self, plaintext):
        ciphertext = DeterministicIterativeAffineCiphertext(plaintext,
                                                            self.n_array[-1],
                                                            self.affine_encoder.mult)
        for i in range(self.key_round):
            ciphertext = self.raw_encrypt_round(ciphertext, i)
        return ciphertext

    def raw_decrypt(self, ciphertext):
        plaintext = ciphertext.cipher
        for i in range(self.key_round):
            plaintext = self.raw_decrypt_round(plaintext, i)
        return plaintext

    def raw_encrypt_round(self, plaintext, round_index):
        return DeterministicIterativeAffineCiphertext(
            (self.a_array[round_index] * plaintext.cipher) % self.n_array[round_index],
            plaintext.n_final,
            self.affine_encoder.mult
        )

    def raw_decrypt_round(self, ciphertext, round_index):
        plaintext = int((mpz(self.a_inv_array[self.key_round - 1 - round_index]) * ciphertext)
                        % self.n_array[self.key_round - 1 - round_index])

        if plaintext / self.n_array[self.key_round - 1 - round_index] > 0.9:
            return plaintext - self.n_array[self.key_round - 1 - round_index]
        else:
            return plaintext


class IterativeAffineCiphertext(object):
    def __init__(self, n_final, multiple, mult_times):
        self.n_final = n_final
        self.multiple = multiple
        self.mult_times = mult_times


class RandomizedIterativeAffineCiphertext(IterativeAffineCiphertext):
    def __init__(self, cipher1, cipher2, n_final, multiple=2 ** 23, mult_times=0):
        super(RandomizedIterativeAffineCiphertext, self).__init__(n_final, multiple, mult_times)
        self.cipher1 = cipher1
        self.cipher2 = cipher2

    def __add__(self, other):
        if isinstance(other, RandomizedIterativeAffineCiphertext):
            if self.multiple != other.multiple or self.n_final != other.n_final:
                raise TypeError("Two addends must have equal multiples and n_finals")
            if self.mult_times > other.mult_times:
                mult_times_diff = self.mult_times - other.mult_times
                return RandomizedIterativeAffineCiphertext(
                    cipher1=(self.cipher1 + other.cipher1 * other.multiple * mult_times_diff) % self.n_final,
                    cipher2=(self.cipher2 + other.cipher2 * other.multiple * mult_times_diff) % self.n_final,
                    n_final=self.n_final,
                    multiple=self.multiple,
                    mult_times=self.mult_times
                )
            elif self.mult_times < other.mult_times:
                mult_times_diff = other.mult_times - self.mult_times
                return RandomizedIterativeAffineCiphertext(
                    cipher1=(other.cipher1 + self.cipher1 * self.multiple * mult_times_diff) % self.n_final,
                    cipher2=(other.cipher2 + self.cipher2 * self.multiple * mult_times_diff) % self.n_final,
                    n_final=self.n_final,
                    multiple=self.multiple,
                    mult_times=other.mult_times
                )
            else:
                return RandomizedIterativeAffineCiphertext(
                    cipher1=(self.cipher1 + other.cipher1) % self.n_final,
                    cipher2=(self.cipher2 + other.cipher2) % self.n_final,
                    n_final=self.n_final,
                    multiple=self.multiple,
                    mult_times=self.mult_times
                )
        elif isinstance(other, int) and other == 0:
            return self
        else:
            raise TypeError("Addition only supports RandomizedIterativeAffineCiphertext"
                            "and initialization with int zero")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, np.float32) or isinstance(other, np.float64):
            return RandomizedIterativeAffineCiphertext(
                cipher1=self.cipher1 * int(other * self.multiple) % self.n_final,
                cipher2=self.cipher2 * int(other * self.multiple) % self.n_final,
                n_final=self.n_final,
                multiple=self.multiple,
                mult_times=self.mult_times + 1
            )
        elif isinstance(other, int) or isinstance(other, np.int32) or isinstance(other, np.int64):
            return RandomizedIterativeAffineCiphertext(
                cipher1=self.cipher1 * int(other) % self.n_final,
                cipher2=self.cipher2 * int(other) % self.n_final,
                n_final=self.n_final,
                multiple=self.multiple,
                mult_times=self.mult_times
            )
        else:
            raise TypeError("Multiplication only supports native and numpy int and float")

    def __rmul__(self, other):
        return self.__mul__(other)


class DeterministicIterativeAffineCiphertext(IterativeAffineCiphertext):
    def __init__(self, cipher, n_final, multiple=2 ** 23, mult_times=0):
        super(DeterministicIterativeAffineCiphertext, self).__init__(n_final, multiple, mult_times)
        self.cipher = cipher

    def __add__(self, other):
        if isinstance(other, DeterministicIterativeAffineCiphertext):
            if self.multiple != other.multiple or self.n_final != other.n_final:
                raise TypeError("Two addends must have equal multiples and n_finals")
            if self.mult_times > other.mult_times:
                mult_times_diff = self.mult_times - other.mult_times
                return DeterministicIterativeAffineCiphertext(
                    cipher=(self.cipher + other.cipher * other.multiple * mult_times_diff) % self.n_final,
                    n_final=self.n_final,
                    multiple=self.multiple,
                    mult_times=self.mult_times
                )
            elif self.mult_times < other.mult_times:
                mult_times_diff = other.mult_times - self.mult_times
                return DeterministicIterativeAffineCiphertext(
                    cipher=(self.cipher * self.multiple * mult_times_diff + other.cipher) % self.n_final,
                    n_final=self.n_final,
                    multiple=self.multiple,
                    mult_times=other.mult_times
                )
            else:
                return DeterministicIterativeAffineCiphertext(
                    cipher=(self.cipher + other.cipher) % self.n_final,
                    n_final=self.n_final,
                    multiple=self.multiple,
                    mult_times=other.mult_times
                )
        elif isinstance(other, int) and other == 0:
            return self
        else:
            raise TypeError("Addition only supports IterativeAffineCiphertext and initialization with int zero")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, np.float32) or isinstance(other, np.float64):
            return DeterministicIterativeAffineCiphertext(
                cipher=self.cipher * int(other * self.multiple) % self.n_final,
                n_final=self.n_final,
                multiple=self.multiple,
                mult_times=self.mult_times + 1
            )
        elif isinstance(other, int) or isinstance(other, np.int32) or isinstance(other, np.int64):
            return DeterministicIterativeAffineCiphertext(
                cipher=self.cipher * int(other) % self.n_final,
                n_final=self.n_final,
                multiple=self.multiple,
                mult_times=self.mult_times
            )
        else:
            raise TypeError("Multiplication only supports native and numpy int and float")

    def __rmul__(self, other):
        return self.__mul__(other)
