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


from fate_crypto.psi import Curve25519


class EllipticCurve(object):
    """
    Symmetric encryption key
    """

    def __init__(self, curve_name, curve_key=None):
        self.curve = self.__get_curve_instance(curve_name, curve_key)

    @staticmethod
    def __get_curve_instance(curve_name, curve_key):
        if curve_key is None:
            return Curve25519()
        return Curve25519(curve_key)

    def get_curve_key(self):
        return self.curve.get_private_key()

    def encrypt(self, plaintext):
        """
        Encryption method
        :param plaintext:
        :return:
        """
        return self.curve.encrypt(plaintext)

    def sign(self, ciphertext):
        return self.curve.diffie_hellman(ciphertext)

    def map_hash_encrypt(self, plaintable, mode, hash_operator, salt):
        """
        adapted from CryptorExecutor
        Process the input Table as (k, v)
        (k, enc_k) for mode == 0
        (enc_k, -1) for mode == 1
        (enc_k, v) for mode == 2
        (k, (enc_k, v)) for mode == 3
        (enc_k, k) for mode == 4
        (enc_k, (k, v)) for mode == 5
        :param plaintable: Table
        :param mode: int
        :return: Table
        """
        if mode == 0:
            return plaintable.map(
                lambda k, v: (
                    k, self.curve.encrypt(
                        hash_operator.compute(
                            k, suffix_salt=salt))))
        elif mode == 1:
            return plaintable.map(
                lambda k, v: (
                    self.curve.encrypt(
                        hash_operator.compute(
                            k, suffix_salt=salt)), -1))
        elif mode == 2:
            return plaintable.map(
                lambda k, v: (
                    self.curve.encrypt(
                        hash_operator.compute(
                            k, suffix_salt=salt)), v))
        elif mode == 3:
            return plaintable.map(
                lambda k, v: (
                    k, (self.curve.encrypt(
                        hash_operator.compute(
                            k, suffix_salt=salt)), v)))
        elif mode == 4:
            return plaintable.map(
                lambda k, v: (
                    self.curve.encrypt(
                        hash_operator.compute(
                            k, suffix_salt=salt)), k))
        elif mode == 5:
            return plaintable.map(
                lambda k, v: (self.curve.encrypt(hash_operator.compute(k, suffix_salt=salt)), (k, v)))

        else:
            raise ValueError("Unsupported mode for elliptic curve map encryption")

    def map_encrypt(self, plaintable, mode):
        """
        adapted from CryptorExecutor
        Process the input Table as (k, v)
        (k, enc_k) for mode == 0
        (enc_k, -1) for mode == 1
        (enc_k, v) for mode == 2
        (k, (enc_k, v)) for mode == 3
        (enc_k, k) for mode == 4
        (enc_k, (k, v)) for mode == 5
        :param plaintable: Table
        :param mode: int
        :return: Table
        """
        if mode == 0:
            return plaintable.map(lambda k, v: (k, self.curve.encrypt(k)))
        elif mode == 1:
            return plaintable.map(lambda k, v: (self.curve.encrypt(k), -1))
        elif mode == 2:
            return plaintable.map(lambda k, v: (self.curve.encrypt(k), v))
        elif mode == 3:
            return plaintable.map(lambda k, v: (k, (self.curve.encrypt(k), v)))
        elif mode == 4:
            return plaintable.map(lambda k, v: (self.curve.encrypt(k), k))
        elif mode == 5:
            return plaintable.map(lambda k, v: (self.curve.encrypt(k), (k, v)))

        else:
            raise ValueError("Unsupported mode for elliptic curve map encryption")

    def map_sign(self, plaintable, mode):
        """
        adapted from CryptorExecutor
        Process the input Table as (k, v)
        (k, enc_k) for mode == 0
        (enc_k, -1) for mode == 1
        (enc_k, v) for mode == 2
        (k, (enc_k, v)) for mode == 3
        (enc_k, k) for mode == 4
        (enc_k, (k, v)) for mode == 5
        :param plaintable: Table
        :param mode: int
        :return: Table
        """
        if mode == 0:
            return plaintable.map(lambda k, v: (k, self.curve.diffie_hellman(k)))
        elif mode == 1:
            return plaintable.map(lambda k, v: (self.curve.diffie_hellman(k), -1))
        elif mode == 2:
            return plaintable.map(lambda k, v: (self.curve.diffie_hellman(k), v))
        elif mode == 3:
            return plaintable.map(lambda k, v: (k, (self.curve.diffie_hellman(k), v)))
        elif mode == 4:
            return plaintable.map(lambda k, v: (self.curve.diffie_hellman(k), k))
        elif mode == 5:
            return plaintable.map(lambda k, v: (self.curve.diffie_hellman(k), (k, v)))

        else:
            raise ValueError("Unsupported mode for elliptic curve map sign")
