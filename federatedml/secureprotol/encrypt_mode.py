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
          'fast': means that only encrypt data in first round, from second round on, just add the difference to
                  previous encrypt data.
          'balance': mixes of 'strict' and 'fast', will re-encrypt all data accords to re_encrypt_rate.
          'confusion_opt": one record use only on confusion in encryption once during iteration.
          'confusion_opt_balance":  balance of 'confusion_opt', will use new confusion according to probability
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
        if self.mode in ["confusion_opt", "confusion_opt_balance"]:
            if self.enc_zeros is None or (
                    self.mode == "confusion_opt_balance" and self.should_re_encrypted()) \
                    or self.enc_zeros.count() != input_data.count():
                self.enc_zeros = input_data.mapValues(lambda val: self.encrypter.encrypt(0))

            new_data = input_data.join(self.enc_zeros, self.add_enc_zero)
            return new_data
        else:
            if self.prev_data is None or self.prev_data.count() != input_data.count() or self.mode == "strict" \
                    or (self.mode == "balance" and self.should_re_encrypted()):
                new_data = input_data.mapValues(self.encrypter.recursive_encrypt)
            else:
                diff_data = input_data.join(self.prev_data, self.get_difference)
                new_data = diff_data.join(self.prev_encrypted_data, self.add_difference)

            self.prev_data = input_data.mapValues(lambda val: val)
            self.prev_encrypted_data = new_data.mapValues(lambda val: val)

            return new_data

    def get_difference(self, new_obj, old_obj):
        """
        Get difference of new_row and old row
        
        Parameters 
        ---------- 
        new_obj: ndarray or single element or iterable python object,
                  
        old_obj: ndarray or single element or iterable python object, data-format should be same with new_data

        Returns 
        ------- 
        diff: ndarray or single element or iterable python object, same data-format of new_row, differance value by new_row subtract old_row

        """
        if isinstance(new_obj, np.ndarray):
            return new_obj - old_obj
        elif isinstance(new_obj, Iterable):
            return type(new_obj)(
                self.get_difference(new_o, old_o) if isinstance(new_o, Iterable) else new_o - old_o for (new_o, old_o)
                in zip(new_obj, old_obj))
        else:
            return new_obj - old_obj

    def add_difference(self, diff_vals, encrypted_data):
        """
        add difference of new_input and old input to previous encrypted_data to get new encrypted_data
        
        Parameters 
        ---------- 
        diff_vals: ndarray or single element or iterable python object,
                  
        encrypted_data: ndarray or single value or iterable python-type, all value in encrypted_data is fate-paillier object

        Returns 
        ------- 
        new_encrypted_data: ndarray or single value or iterable python-type, data-format is same with encrypted_data, equals to sum of encrypted_data and diff_vals
        
        """
        if isinstance(diff_vals, np.ndarray):
            return diff_vals + encrypted_data
        elif isinstance(diff_vals, Iterable):
            return type(diff_vals)(
                self.add_difference(diff_o, enc_o) if isinstance(diff_o, Iterable) else diff_o + enc_o for
                (diff_o, enc_o) in zip(diff_vals, encrypted_data))
        else:
            return diff_vals + encrypted_data
