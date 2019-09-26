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

import unittest

import numpy as np

from arch.api.session import init
from federatedml.ftl.eggroll_computation.helper import distribute_encrypt_matrix, distribute_decrypt_matrix
from federatedml.ftl.test.util import assert_matrix
from federatedml.secureprotol.encrypt import PaillierEncrypt


class TestEncryption(unittest.TestCase):

    def test_encrypt_1_dim(self):
        matrix = np.array([1, 2, 3, 4, 5], dtype=np.float64)

        self.__test(matrix)

    def test_encrypt_2_dim(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9],
                           [10, 11, 12],
                           [13, 14, 15],
                           [16, 17, 18],
                           [19, 20, 21]], dtype=np.float64)

        self.__test(matrix)

    def test_encrypt_3_dim_1(self):
        matrix = np.array([[[33, 22, 31],
                            [14, 15, 16],
                            [17, 18, 19]],
                           [[10, 11, 12],
                            [13, 14, 15],
                            [16, 17, 18]]])

        self.__test(matrix)

    def test_encrypt_3_dim_2(self):
        matrix = np.array([[[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]],
                           [[10, 11, 12],
                            [13, 14, 15],
                            [16, 17, 18]],
                           [[11, 14, 12],
                            [13, 12, 15],
                            [17, 19, 20]]
                           ])

        self.__test(matrix)

    def test_encrypt_3_dim_3(self):
        matrix = np.ones((8, 50, 50))
        self.__test(matrix)

    def __test(self, matrix):
        paillierEncrypt = PaillierEncrypt()
        paillierEncrypt.generate_key()
        publickey = paillierEncrypt.get_public_key()
        privatekey = paillierEncrypt.get_privacy_key()

        result = distribute_encrypt_matrix(publickey, matrix)
        decrypted_result = distribute_decrypt_matrix(privatekey, result)
        assert_matrix(matrix, decrypted_result)


if __name__ == '__main__':
    init()
    unittest.main()
