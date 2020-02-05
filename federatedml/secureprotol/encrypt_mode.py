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
from collections import Iterable

import numpy as np

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
            return obj + enc_zero

    @staticmethod
    def gen_random_number():
        return random.random()

    def should_re_encrypted(self):
        return self.gen_random_number() <= self.re_encrypted_rate + consts.FLOAT_ZERO

    def encrypt(self, input_data):
        """
        Encrypt data according to different mode
        
        Parameters 
        ---------- 
        input_data: DTable

        Returns 
        ------- 
        new_data: DTable, encrypted result of input_data

        """
        if self.mode == "strict":
            new_data = input_data.mapValues(self.encrypter.recursive_encrypt)
            return new_data
        else:
            if self.enc_zeros is None or (
                self.mode == "balance" and self.should_re_encrypted()) \
                    or self.enc_zeros.count() != input_data.count():
                self.enc_zeros = input_data.mapValues(lambda val: self.encrypter.encrypt(0))

            new_data = input_data.join(self.enc_zeros, self.add_enc_zero)
            return new_data


