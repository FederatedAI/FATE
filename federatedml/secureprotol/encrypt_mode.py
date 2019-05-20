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
from federatedml.util import consts
import numpy as np
import random


class EncryptModeCalculator(object):
    def __init__(self, encrypter=None, mode="slow", re_encrypted_rate=1):
        self.encrypter = encrypter
        self.mode = mode
        self.re_encrypted_rate = re_encrypted_rate
        self.prev_data = None
        self.prev_encrypted_data = None
        self.call_round = 0
        self.type_change = {"list": list,
                            "tuple": tuple,
                            "ndarray": np.array}

    def encrypt_row(self, row):
        if isinstance(row, Iterable):
            data_type = type(row).__name__
            if data_type not in self.type_change:
                raise NotImplementedError("data type {} not supported yet".format(data_type))
            else:
                return self.type_change[data_type](self.encrypter.encrypt(val) for val in row)
        else:
            return self.encrypter.encrypt(row)

    def gen_random_number(self):
        return random.random()

    def encrypt(self, input_data):
        self.call_round += 1
        if self.prev_data is None or self.mode == "slow" \
                or (self.mode == "balance" and self.gen_random_number() <= self.re_encrypted_rate + consts.FLOAT_ZERO):
            new_data = input_data.mapValues(self.encrypt_row)
        else:
            new_data = input_data.join(self.prev_data, self.get_differance).join(self.prev_encrypted_data, self.add_differance)

        self.prev_data = input_data
        self.prev_encrypted_data = new_data

        return new_data

    def get_differance(self, new_row, old_row):
        if isinstance(new_row, Iterable):
            diff = [new_val - old_val for (new_val, old_val) in zip(new_row, old_row)]
            if type(new_row).__name__ == "ndarray":
                return np.array(diff)
            else:
                return type(new_row)(diff)
        else:
            return new_row - old_row

    def add_differance(self, diff_vals, encrypted_data):
        if isinstance(diff_vals, Iterable):
            new_encrypted_data = [diff + encrypted_val for (diff, encrypted_val) in zip(diff_vals, encrypted_data)]
            if type(new_encrypted_data).__name__ == "ndarray":
                return np.array(new_encrypted_data)
            else:
                return type(diff_vals)(new_encrypted_data)
        else:
            return diff_vals + encrypted_data

