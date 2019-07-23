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

from collections import Iterable
from federatedml.statistic.data_overview import rubbish_clear 
from federatedml.util import consts
import numpy as np
import random


class EncryptModeCalculator(object):
    """
    Encyprt Mode module, a balance of security level and speed.

    Parameters
    ----------
    encrypter: object, fate-paillier object, object to encrypt numbers

    mode: str, accpet 'strict', 'fast', 'balance'.
          'strict' means that re-encrypted every function call.
          'fast' means that only encrypt data in first round, from second round on, just add the difference to previous encrypt data.
          'balance': mixes of 'strict' and 'fast', will re-encrypt all data accords to re_encrypt_rate.

    re_encrypted_rate: float or int, numeric, use if mode equals to "balance" 
    """

    def __init__(self, encrypter=None, mode="strict", re_encrypted_rate=1):
        self.encrypter = encrypter
        self.mode = mode
        self.re_encrypted_rate = re_encrypted_rate
        self.prev_data = None
        self.prev_encrypted_data = None

    def encrypt_row(self, row):
        if type(row).__name__ == "ndarray":
            return np.array([self.encrypter.encrypt(val) for val in row]) 
        elif isinstance(row, Iterable):
            return type(row)(self.encrypter.encrypt(val) for val in row)
        else:
            return self.encrypter.encrypt(row)

    def gen_random_number(self):
        return random.random()

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
        
        if self.prev_data is None or self.mode == "strict" \
                or (self.mode == "balance" and self.gen_random_number() <= self.re_encrypted_rate + consts.FLOAT_ZERO):
            new_data = input_data.mapValues(self.encrypt_row)
        else:
            diff_data = input_data.join(self.prev_data, self.get_differance) 
            new_data = diff_data.join(self.prev_encrypted_data, self.add_differance)
            
            """temporary code, clear unused table begin"""
            rubbish_list = [diff_data]
            rubbish_clear(rubbish_list)
            """temporary code, clear unused table end"""
            # new_data = input_data.join(self.prev_data, self.get_differance).join(self.prev_encrypted_data, self.add_differance)

        """temporary code, clear unused table begin"""
        rubbish_list = [self.prev_data, 
                        self.prev_encrypted_data]
        rubbish_clear(rubbish_list)
        """temporary code, clear unused table end"""

        self.prev_data = input_data.mapValues(lambda val: val)
        self.prev_encrypted_data = new_data.mapValues(lambda val: val)

        return new_data

    def get_differance(self, new_row, old_row):
        """
        Get difference of new_row and old row
        
        Parameters 
        ---------- 
        new_row: ndarray or single element or iterable python object,
                  
        old_row: ndarray or single element or iterable python object, data-format should be same with new_data

        Returns 
        ------- 
        diff: ndarray or single element or iterable python object, same data-format of new_row, differance value by new_row subtract old_row

        """

        if type(new_row).__name__ == "ndarray":
            diff = [new_val - old_val for (new_val, old_val) in zip(new_row, old_row)]
            return np.array(diff)
        elif isinstance(new_row, Iterable):
            diff = [new_val - old_val for (new_val, old_val) in zip(new_row, old_row)]
            return type(new_row)(diff)
        else:
            return new_row - old_row

    def add_differance(self, diff_vals, encrypted_data):
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
        
        if type(diff_vals).__name__ == "ndarray":
            new_encrypted_data = [diff + encrypted_val for (diff, encrypted_val) in zip(diff_vals, encrypted_data)]
            return np.array(new_encrypted_data)
        elif isinstance(diff_vals, Iterable):
            new_encrypted_data = [diff + encrypted_val for (diff, encrypted_val) in zip(diff_vals, encrypted_data)]
            return type(diff_vals)(new_encrypted_data)
        else:
            return diff_vals + encrypted_data

