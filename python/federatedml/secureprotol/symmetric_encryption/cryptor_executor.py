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

import functools


class CryptoExecutor(object):
    def __init__(self, cipher_core):
        self.cipher_core = cipher_core

    def init(self):
        self.cipher_core.init()

    def renew(self, cipher_core):
        self.cipher_core = cipher_core

    def map_hash_encrypt(self, plaintable, mode, hash_operator, salt):
        """
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
                    k, self.cipher_core.encrypt(
                        hash_operator.compute(
                            k, suffix_salt=salt))))
        elif mode == 1:
            return plaintable.map(
                lambda k, v: (
                    self.cipher_core.encrypt(
                        hash_operator.compute(
                            k, suffix_salt=salt)), -1))
        elif mode == 2:
            return plaintable.map(
                lambda k, v: (
                    self.cipher_core.encrypt(
                        hash_operator.compute(
                            k, suffix_salt=salt)), v))
        elif mode == 3:
            return plaintable.map(
                lambda k, v: (
                    k, (self.cipher_core.encrypt(
                        hash_operator.compute(
                            k, suffix_salt=salt)), v)))
        elif mode == 4:
            return plaintable.map(
                lambda k, v: (
                    self.cipher_core.encrypt(
                        hash_operator.compute(
                            k, suffix_salt=salt)), k))
        elif mode == 5:
            return plaintable.map(
                lambda k, v: (self.cipher_core.encrypt(hash_operator.compute(k, suffix_salt=salt)), (k, v)))

        else:
            raise ValueError("Unsupported mode for crypto_executor map encryption")

    def map_encrypt(self, plaintable, mode):
        """
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
            return plaintable.map(lambda k, v: (k, self.cipher_core.encrypt(k)))
        elif mode == 1:
            return plaintable.map(lambda k, v: (self.cipher_core.encrypt(k), -1))
        elif mode == 2:
            return plaintable.map(lambda k, v: (self.cipher_core.encrypt(k), v))
        elif mode == 3:
            return plaintable.map(lambda k, v: (k, (self.cipher_core.encrypt(k), v)))
        elif mode == 4:
            return plaintable.map(lambda k, v: (self.cipher_core.encrypt(k), k))
        elif mode == 5:
            return plaintable.map(lambda k, v: (self.cipher_core.encrypt(k), (k, v)))

        else:
            raise ValueError("Unsupported mode for crypto_executor map encryption")

    def map_values_encrypt(self, plaintable, mode):
        """
        Process the input Table as v
        enc_v if mode == 0
        :param plaintable: Table
        :param mode: int
        :return:
        """
        if mode == 0:
            return plaintable.mapValues(lambda v: self.cipher_core.encrypt(v))
        else:
            raise ValueError("Unsupported mode for crypto_executor map_values encryption")

    def map_decrypt(self, ciphertable, mode):
        """
        Process the input Table as (k, v)
        (k, dec_k) for mode == 0
        (dec_k, -1) for mode == 1
        (dec_k, v) for mode == 2
        (k, (dec_k, v)) for mode == 3
        :param ciphertable: Table
        :param mode: int
        :return: Table
        """
        if mode == 0:
            return ciphertable.map(lambda k, v: (k, self.cipher_core.decrypt(k)))
        elif mode == 1:
            return ciphertable.map(lambda k, v: (self.cipher_core.decrypt(k), -1))
        elif mode == 2:
            return ciphertable.map(lambda k, v: (self.cipher_core.decrypt(k), v))
        elif mode == 3:
            return ciphertable.map(lambda k, v: (k, (self.cipher_core.decrypt(k), v)))
        elif mode == 4:
            return ciphertable.map(lambda k, v: (self.cipher_core.decrypt(k), v))
        elif mode == 5:
            return ciphertable.map(lambda k, v: (self.cipher_core.decrypt(k), v))
        else:
            raise ValueError("Unsupported mode for crypto_executor map decryption")

    def map_values_decrypt(self, ciphertable, mode):
        """
        Process the input Table as v
        dec_v if mode == 0
        decode(dec_v) if mode == 1
        :param ciphertable: Table
        :param mode: int
        :return:
        """
        if mode == 0:
            return ciphertable.mapValues(lambda v: self.cipher_core.decrypt(v))
        elif mode == 1:
            f = functools.partial(self.cipher_core.decrypt, decode_output=True)
            return ciphertable.mapValues(lambda v: f(v))
        else:
            raise ValueError("Unsupported mode for crypto_executor map_values encryption")

    def get_nonce(self):
        return self.cipher_core.get_nonce()
