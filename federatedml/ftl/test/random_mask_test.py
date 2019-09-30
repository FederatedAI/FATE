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
from federatedml.ftl.data_util.common_data_util import add_random_mask_for_list_of_values, add_random_mask, \
    remove_random_mask_from_list_of_values, remove_random_mask
from federatedml.ftl.eggroll_computation.helper import distribute_encrypt_matrix, distribute_decrypt_matrix
from federatedml.ftl.test.util import assert_matrix
from federatedml.secureprotol.encrypt import PaillierEncrypt


class TestRandomMask(unittest.TestCase):

    def test_mask_2_dim_matrix(self):
        print("----test_mask_2_dim----")
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9],
                           [10, 11, 12],
                           [13, 14, 15],
                           [16, 17, 18],
                           [19, 20, 21]], dtype=np.float64)

        self.__test_matrix(matrix)

    def test_mask_3_dim_matrix_1(self):
        print("----test_mask_3_dim_1----")
        matrix = np.array([[[33, 22, 31],
                            [14, 15, 16],
                            [17, 18, 19]],
                           [[10, 11, 12],
                            [13, 14, 15],
                            [16, 17, 18]]])

        self.__test_matrix(matrix)

    def test_mask_3_dim_matrix_2(self):
        print("----test_encrypt_3_dim_2----")
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

        self.__test_matrix(matrix)

    def test_mask_integer(self):
        print("----test_mask_integer----")
        value = 31
        self.__test_scalar(value)

    def test_mask_float(self):
        print("----test_mask_float----")
        value = 31.444
        self.__test_scalar(value)

    def test_mask_raise_type_error(self):
        print("----test_mask_scalar----")
        value = "exception"
        with self.assertRaises(TypeError):
            self.__test_scalar(value)

    def test_mask_list_of_values(self):
        matrix_1 = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9],
                             [10, 11, 12],
                             [13, 14, 15],
                             [16, 17, 18],
                             [19, 20, 21]])

        matrix_2 = np.array([[[33, 22, 31],
                              [14, 15, 16],
                              [17, 18, 19]],
                             [[10, 11, 12],
                              [13, 14, 15],
                              [16, 17, 18]]])

        matrix_list = [matrix_1, matrix_2]
        masked_value_list, mask_list = add_random_mask_for_list_of_values(matrix_list)
        cleared_value_list = remove_random_mask_from_list_of_values(masked_value_list, mask_list)

        for matrix, cleared_matrix in zip(matrix_list, cleared_value_list):
            assert_matrix(matrix, cleared_matrix)

    def __test_matrix(self, matrix):

        paillierEncrypt = PaillierEncrypt()
        paillierEncrypt.generate_key()
        publickey = paillierEncrypt.get_public_key()
        privatekey = paillierEncrypt.get_privacy_key()

        enc_matrix = distribute_encrypt_matrix(publickey, matrix)
        masked_enc_matrix, mask = add_random_mask(enc_matrix)

        cleared_enc_matrix = remove_random_mask(masked_enc_matrix, mask)
        cleared_matrix = distribute_decrypt_matrix(privatekey, cleared_enc_matrix)
        assert_matrix(matrix, cleared_matrix)

    def __test_scalar(self, value):

        paillierEncrypt = PaillierEncrypt()
        paillierEncrypt.generate_key()
        publickey = paillierEncrypt.get_public_key()
        privatekey = paillierEncrypt.get_privacy_key()

        enc_value = publickey.encrypt(value)
        masked_enc_value, mask = add_random_mask(enc_value)

        cleared_enc_value = remove_random_mask(masked_enc_value, mask)
        cleared_value = privatekey.decrypt(cleared_enc_value)
        print("original matrix", value)
        print("cleared_matrix", cleared_value)
        self.assertEqual(value, cleared_value)


if __name__ == '__main__':
    init()
    unittest.main()



