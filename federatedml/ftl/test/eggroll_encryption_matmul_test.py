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
from federatedml.ftl.eggroll_computation.helper import distribute_encrypt_matmul_2_ob, distribute_encrypt_matmul_3
from federatedml.ftl.encryption.encryption import decrypt_matrix
from federatedml.ftl.test.util import assert_matrix
from federatedml.secureprotol import PaillierEncrypt


class TestEncryptionMatmul(unittest.TestCase):

    def setUp(self):
        paillierEncrypt = PaillierEncrypt()
        paillierEncrypt.generate_key()
        self.publickey = paillierEncrypt.get_public_key()
        self.privatekey = paillierEncrypt.get_privacy_key()

    def encrypt_2d_matrix(self, X):
        encrypt_X = [[0 for _ in range(X.shape[1])] for _ in range(X.shape[0])]
        for i in range(len(encrypt_X)):
            temp = []
            for j in range(X.shape[-1]):
                temp.append(self.publickey.encrypt(X[i, j]))
            encrypt_X[i] = temp

        encrypt_X = np.array(encrypt_X)
        return encrypt_X

    def encrypt_3d_matrix(self, X):
        encrypt_X = [[[0 for _ in range(X.shape[-1])] for _ in range(X.shape[1])] for _ in range(X.shape[0])]
        for i in range(X.shape[0]):
            second_dim_list = []
            for j in range(X.shape[1]):
                third_dim_list = []
                for z in range(X.shape[-1]):
                    third_dim_list.append(self.publickey.encrypt(X[i, j, z]))
                second_dim_list.append(third_dim_list)
            encrypt_X[i] = second_dim_list
        return np.array(encrypt_X)

    def test_encrypt_matmul_2_dim(self):

        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=np.float64)
        Y = np.array([[10, 11, 12],
                      [13, 14, 15],
                      [16, 17, 18]], dtype=np.float64)

        Z = np.matmul(X, Y)

        encrypt_Y = self.encrypt_2d_matrix(Y)
        res = distribute_encrypt_matmul_2_ob(X, encrypt_Y)

        # decrypt res
        decrypt_res = decrypt_matrix(self.privatekey, res)
        assert_matrix(Z, decrypt_res)

    def test_encrypt_matmul_3_dim_1(self):

        X = np.array([[[1, 2, 3]],
                      [[10, 11, 12]]], dtype=np.float64)
        Y = np.array([[[10, 11, 12],
                       [13, 14, 15],
                       [16, 17, 18]],
                      [[19, 20, 21],
                       [22, 23, 24],
                       [25, 26, 27]]], dtype=np.float64)

        Z = np.matmul(X, Y)

        encrypt_Y = self.encrypt_3d_matrix(Y)

        res = distribute_encrypt_matmul_3(X, encrypt_Y)

        # decrypt res
        decrypt_res = decrypt_matrix(self.privatekey, res)
        assert_matrix(Z, decrypt_res)

    def test_encrypt_matmul_3_dim_2(self):

        X = np.array([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]],
                      [[10, 11, 12],
                       [13, 14, 15],
                       [16, 17, 18]]], dtype=np.float64)
        Y = np.array([[[10, 11, 12],
                       [13, 14, 15],
                       [16, 17, 18]],
                      [[19, 20, 21],
                       [22, 23, 24],
                       [25, 26, 27]]], dtype=np.float64)

        Z = np.matmul(X, Y)

        encrypt_Y = self.encrypt_3d_matrix(Y)
        res = distribute_encrypt_matmul_3(X, encrypt_Y)

        decrypt_res = decrypt_matrix(self.privatekey, res)
        assert_matrix(Z, decrypt_res)

    def test_encrypt_matmul_3_dim_3(self):

        X = np.array([[[1, 2, 3]],
                      [[10, 11, 12]]], dtype=np.float64)
        Y = np.array([[[10, 11, 12],
                       [13, 14, 15],
                       [16, 17, 18]],
                      [[19, 20, 21],
                       [22, 23, 24],
                       [25, 26, 27]]], dtype=np.float64)

        Z = np.matmul(X, Y)

        encrypt_Y = self.encrypt_3d_matrix(Y)
        res = distribute_encrypt_matmul_3(X, encrypt_Y)

        decrypt_res = decrypt_matrix(self.privatekey, res)
        assert_matrix(Z, decrypt_res)


if __name__ == '__main__':
    init()
    unittest.main()
