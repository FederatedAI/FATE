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

import copy
import numpy as np
import unittest
from arch.api import eggroll
from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator
from federatedml.secureprotol import PaillierEncrypt


class TestEncryptModeCalculator(unittest.TestCase):
    def setUp(self):
        eggroll.init("test_encrypt_mode_calculator")

        self.list_data = []
        self.tuple_data = []
        self.numpy_data = []

        for i in range(30):
            list_value = [100 * i + j for j in range(20)]
            tuple_value = tuple(list_value)
            numpy_value = np.array(list_value, dtype="int")

            self.list_data.append(list_value)
            self.tuple_data.append(tuple_value)
            self.numpy_data.append(numpy_value)

        self.data_list = eggroll.parallelize(self.list_data, include_key=False, partition=10)
        self.data_tuple = eggroll.parallelize(self.tuple_data, include_key=False, partition=10)
        self.data_numpy = eggroll.parallelize(self.numpy_data, include_key=False, partition=10)
       
    def test_data_type(self, mode="strict", re_encrypted_rate=0.2):
        encrypter = PaillierEncrypt()
        encrypter.generate_key(1024)
        encrypted_calculator = EncryptModeCalculator(encrypter, mode, re_encrypted_rate)        

        data_list = dict(encrypted_calculator.encrypt(self.data_list).collect())
        data_tuple = dict(encrypted_calculator.encrypt(self.data_tuple).collect())
        data_numpy = dict(encrypted_calculator.encrypt(self.data_numpy).collect())
        
        for key, value in data_list.items():
            self.assertTrue(isinstance(value, list))
            self.assertTrue(len(value) == len(self.list_data[key]))
        
        for key, value in data_tuple.items():
            self.assertTrue(isinstance(value, tuple))
            self.assertTrue(len(value) == len(self.tuple_data[key]))

        for key, value in data_numpy.items():
            self.assertTrue(type(value).__name__ == "ndarray")
            self.assertTrue(value.shape[0] == self.numpy_data[key].shape[0])

    def test_data_type_with_diff_mode(self):
        self.test_data_type(mode="strict")
        self.test_data_type(mode="fast")
        self.test_data_type(mode="balance")

    def test_diff_mode(self, round=10, mode="strict", re_encrypted_rate=0.2):
        encrypter = PaillierEncrypt()
        encrypter.generate_key(1024)
        encrypted_calculator = EncryptModeCalculator(encrypter, mode, re_encrypted_rate)        

        for i in range(round):
            data_i = self.data_numpy.mapValues(lambda v: v + i)
            data_i = encrypted_calculator.encrypt(data_i)
            decrypt_data_i = dict(data_i.mapValues(lambda arr: np.array([encrypter.decrypt(val) for val in arr])).collect())
            for j in range(30):
                self.assertTrue(np.fabs(self.numpy_data[j] - decrypt_data_i[j] + i).all() < 1e-5)
           
    def test_balance_mode(self):
        self.test_diff_mode(mode="strict")
        self.test_diff_mode(mode="fast")
        self.test_diff_mode(mode="balance")


if __name__ == '__main__':
    unittest.main()
