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
from federatedml.secureprotol.symmetric_encryption.symmetric_encryption import SymmetricKey, SymmetricCiphertext
from federatedml.util import conversion


class XorCipherKey(SymmetricKey):
    """
    key = (self.alpha, self.beta), expected to be 256 bits
    Enc(m) = (m XOR self.alpha, self.beta)
    Dec(c) = c.message XOR alpha if c.verifier == self.beta, None otherwise
    Note that the maximum size of the plaintext supported is principally determined by len(key) // 2
    """

    def __init__(self, key):
        """
        self.alpha and self.beta are str-typed binaries, e.g., '1010'
        :param key: bytes
        """
        super(XorCipherKey, self).__init__()
        self.alpha = conversion.bytes_to_bin(key[:(len(key) // 2)])     # binary string
        self.beta = conversion.bytes_to_bin(key[(len(key) // 2):])      # binary string
        if len(self.beta) % 8 != 0:
            raise ValueError("XOR encryption invalid key")
        self.beta_string = conversion.bin_to_str(self.beta)             # unicode-string

    def encrypt(self, plaintext):
        """

        :param plaintext: int/float/str
        :return: XorCiphertext
        """
        plaintext_bin = self._all_to_bin(plaintext)
        if plaintext_bin == -1:
            raise TypeError('Xor encryption only supports int/float/str plaintext')
        ciphertext_bin = self._xor(plaintext_bin, self.alpha)
        ciphertext = self._bin_to_str(ciphertext_bin)
        return XorCiphertext(ciphertext, self.beta_string[:len(ciphertext)])

    def decrypt(self, ciphertext):
        """

        :param ciphertext: XorCiphertext
        :return: str
        """
        if ciphertext.verifier != self.beta_string[:len(ciphertext.verifier)]:
            raise ValueError("XOR encryption invalid ciphertext")
        ciphertext_bin = self._all_to_bin(ciphertext.message)
        plaintext_bin = self._xor(ciphertext_bin, self.alpha)
        return self._bin_to_str(plaintext_bin)

    @staticmethod
    def _xor(str1, str2):
        """
        Compute the bit-wise XOR result of two binary numbers in string, e.g., 01011010 = _xor('10101010', '11110010')
        If two string are different in length, XOR starts applying from highest (left-most) bit, and abandons the longer
            one's mantissa
        :param str1: str, whose length must be a multiple of 8
        :param str2: str, whose length must be a multiple of 8
        :return: str, whose length must be a multiple of 8
        """
        res = ''
        for i in range(min(len(str1), len(str2))):
            res += XorCipherKey._xor_bit(str1[i], str2[i])
        return res

    @staticmethod
    def _xor_bit(char1, char2):
        """
        Compute the XOR result of two bits in string, e.g., '1' = _xor_bit('0', '1')
        :param char1: str
        :param char2: str
        :return: str
        """
        return '0' if char1 == char2 else '1'

    @staticmethod
    def _all_to_bin(message):
        """
        Convert an int/float/str to a binary number in string, e.g., 1.65 -> '110001101110110110110101'
        :param message: int/float/str
        :return: -1 if type error, otherwise str
        """
        if isinstance(message, int) or isinstance(message, float):
            return conversion.str_to_bin(str(message))
        elif isinstance(message, str):
            return conversion.str_to_bin(message)
        else:
            return -1

    @staticmethod
    def _bin_to_str(message):
        """
        Convert a binary number in string to Unicode string
        :param message: str, whose length must be a multiple of 8
        :return: str
        """
        return conversion.bin_to_str(message)


class XorCiphertext(SymmetricCiphertext):
    """
    ciphertext = (self.message, self.verifier)
    """

    def __init__(self, message, verifier):
        super(XorCiphertext, self).__init__()
        self.message = message
        self.verifier = verifier
