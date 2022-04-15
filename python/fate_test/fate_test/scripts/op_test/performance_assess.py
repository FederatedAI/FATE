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

import time
import numpy as np


# Operations
class Metric(object):
    def __init__(self, data_num, test_round):
        self.operation = None
        self.data_num = data_num
        self.test_round = test_round

    @staticmethod
    def accuracy(rand_data, decrypt_data):
        difference = 0
        for x, y in zip(rand_data, decrypt_data):
            difference += abs(abs(x) - abs(y))
        abs_acc = abs(difference) / len(rand_data)
        difference = 0
        for x, y in zip(rand_data, decrypt_data):
            difference += abs(abs(x) - abs(y)) / (1e-100 + max(abs(x), abs(y)))
        relative_acc = difference / len(rand_data)
        log_acc = -np.log2(relative_acc) if relative_acc != 0 else 0

        return abs_acc, relative_acc, log_acc

    @staticmethod
    def many_call(data_x, unary_op=None, binary_op=None, data_y=None, test_round=1):
        if unary_op is not None:
            time_start = time.time()
            for _ in range(test_round):
                _ = list(map(unary_op, data_x))
            final_time = time.time() - time_start
        else:
            time_start = time.time()
            for _ in range(test_round):
                _ = list(map(binary_op, data_x, data_y))
            final_time = time.time() - time_start

        return final_time / test_round

    def encrypt(self, data, op):
        many_round_encrypt_time = self.many_call(data, unary_op=op, test_round=self.test_round)
        single_encrypt_time = many_round_encrypt_time / self.data_num
        cals_per_second = self.data_num / many_round_encrypt_time

        return ["encrypt", '%.10f' % single_encrypt_time + 's', '%.10f' % many_round_encrypt_time + 's', "-", "-",
                int(cals_per_second), "-"]

    def decrypt(self, encrypt_data, data, decrypt_data, function):
        many_round_decrypt_time = self.many_call(encrypt_data, function, test_round=self.test_round)
        single_decrypt_time = many_round_decrypt_time / self.data_num
        cals_per_second = self.data_num / many_round_decrypt_time
        abs_acc, relative_acc, log_acc = self.accuracy(data, decrypt_data)
        return ["decrypt", '%.10f' % single_decrypt_time + 's', '%.10f' % many_round_decrypt_time + 's',
                relative_acc, log_acc, int(cals_per_second), "-"]

    def binary_op(self, encrypt_data_x, encrypt_data_y,
                  raw_data_x, raw_data_y, real_ret, decrypt_ret, op, op_name):
        many_round_time = self.many_call(data_x=encrypt_data_x, binary_op=op,
                                         data_y=encrypt_data_y, test_round=self.test_round)
        single_op_time = many_round_time / self.data_num
        cals_per_second = self.data_num / many_round_time

        plaintext_per_second = self.data_num / self.many_call(data_x=raw_data_x, data_y=raw_data_y,
                                                              binary_op=op, test_round=self.test_round)

        abs_acc, relative_acc, log_acc = self.accuracy(real_ret, decrypt_ret)
        return [op_name, '%.10f' % single_op_time + 's', '%.10f' % many_round_time + 's',
                relative_acc, log_acc, int(cals_per_second), int(plaintext_per_second)]
