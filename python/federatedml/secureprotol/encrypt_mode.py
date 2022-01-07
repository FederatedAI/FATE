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
import functools
import numpy as np
from collections import Iterable
from federatedml.secureprotol import PaillierEncrypt
from federatedml.util import consts


class EncryptModeCalculator(object):
    """
    Encyprt Mode module, a balance of security level and speed.

    Parameters
    ----------
    encrypter: object, fate-paillier object, object to encrypt numbers

    mode: str, accpet 'strict', 'fast', 'balance'. "confusion_opt", "confusion_opt_balance"
          'strict': means that re-encrypted every function call.
          'fast/confusion_opt": one record use only on confusion in encryption once during iteration.
          'balance/confusion_opt_balance":  balance of 'confusion_opt', will use new confusion according to probability
                                    decides by 're_encrypted_rate'
    re_encrypted_rate: float or float, numeric, use if mode equals to "balance" or "confusion_opt_balance"

    """

    def __init__(self, encrypter=None, mode="strict", re_encrypted_rate=1):
        self.encrypter = encrypter
        self.mode = mode
        self.re_encrypted_rate = re_encrypted_rate
        self.prev_data = None
        self.prev_encrypted_data = None
        self.enc_zeros = None

        self.align_to_input_data = True
        self.soft_link_mode()

    def soft_link_mode(self):

        if self.mode == "strict":
            return

        if self.mode in ["confusion_opt", "fast"]:
            self.mode = "fast"

        if self.mode in ["confusion_opt_balance", "balance"]:
            self.mode = "balance"

    @staticmethod
    def add_enc_zero(obj, enc_zero):
        if isinstance(obj, np.ndarray):
            return obj + enc_zero
        elif isinstance(obj, Iterable):
            return type(obj)(
                EncryptModeCalculator.add_enc_zero(o, enc_zero) if isinstance(o, Iterable) else o + enc_zero for o in
                obj)
        else:
            return enc_zero + obj

    @staticmethod
    def gen_random_number():
        return random.random()

    def should_re_encrypted(self):
        return self.gen_random_number() <= self.re_encrypted_rate + consts.FLOAT_ZERO

    def set_enc_zeros(self, input_data, enc_func):
        self.enc_zeros = input_data.mapValues(lambda val: enc_func(0))

    def re_encrypt(self, input_data, enc_func):
        if input_data is None:  # no need to re-encrypt
            return
        self.set_enc_zeros(input_data, enc_func)

    def encrypt_data(self, input_data, enc_func):

        if self.mode == "strict":
            new_data = input_data.mapValues(enc_func)
            return new_data
        else:
            target_data = input_data
            if self.enc_zeros is None:
                self.set_enc_zeros(target_data, enc_func)
            elif self.mode == "balance" and self.should_re_encrypted():
                if not self.align_to_input_data:
                    target_data = self.enc_zeros
                self.re_encrypt(target_data, enc_func)
            elif self.enc_zeros.count() != input_data.count():
                if not self.align_to_input_data:
                    target_data = None
                self.re_encrypt(target_data, enc_func)
            new_data = input_data.join(self.enc_zeros, self.add_enc_zero)
            return new_data

    def get_enc_func(self, encrypter, raw_enc=False, exponent=0):

        if not raw_enc:
            return encrypter.recursive_encrypt
        else:
            if type(self.encrypter) == PaillierEncrypt:
                raw_en_func = functools.partial(self.encrypter.recursive_raw_encrypt, exponent=exponent)
            else:
                raw_en_func = self.encrypter.recursive_raw_encrypt

            return raw_en_func

    def encrypt(self, input_data):
        """
        Encrypt data according to different mode
        
        Parameters 
        ---------- 
        input_data: Table

        Returns 
        ------- 
        new_data: Table, encrypted result of input_data

        """
        encrypt_func = self.get_enc_func(self.encrypter, raw_enc=False)
        new_data = self.encrypt_data(input_data, encrypt_func)
        return new_data

    def raw_encrypt(self, input_data, exponent=0):

        raw_en_func = self.get_enc_func(self.encrypter, raw_enc=True, exponent=exponent)
        new_data = self.encrypt_data(input_data, raw_en_func)
        return new_data

    def init_enc_zero(self, input_data, raw_en=False, exponent=0):
        en_func = self.get_enc_func(self.encrypter, raw_en, exponent)
        self.set_enc_zeros(input_data, en_func)

    def recursive_encrypt(self, input_data):
        return self.encrypter.recursive_encrypt(input_data)

    def distribute_encrypt(self, input_data):
        return self.encrypt(input_data)

    def distribute_decrypt(self, input_data):
        return self.encrypter.distribute_decrypt(input_data)

    def recursive_decrypt(self, input_data):
        return self.encrypter.recursive_decrypt(input_data)
