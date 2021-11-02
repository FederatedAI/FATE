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

import numpy as np
from federatedml.secureprotol.paillier_tensor import PaillierTensor
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator
from federatedml.param.encrypt_param import EncryptParam
from federatedml.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from federatedml.nn.hetero_nn.util import random_number_generator
import unittest
from federatedml.util import consts
from fate_arch.session import computing_session as session


class TestPaillierTensor(unittest.TestCase):

    def setUp(self):
        session.init('test', 0)

    def test_tensor_op(self):

        arr1 = np.ones((10, 1, 3))
        arr1[0] = np.array([[2, 3, 4]])
        arr2 = np.ones((10, 3, 3))
        arr3 = np.ones([1, 1, 3])

        arr4 = np.ones([50, 1])
        arr5 = np.ones([32])

        pt = PaillierTensor(arr1)
        pt2 = PaillierTensor(arr2)
        pt3 = PaillierTensor(arr3)

        pt4 = PaillierTensor(arr4)
        pt5 = PaillierTensor(arr5)

        encrypter = PaillierEncrypt()
        encrypter.generate_key(EncryptParam().key_length)
        encrypted_calculator = EncryptModeCalculator(encrypter,
                                                     EncryptedModeCalculatorParam().mode,
                                                     EncryptedModeCalculatorParam().re_encrypted_rate)
        rs1 = pt * arr2
        rs2 = pt * pt2
        rs3 = pt.matmul_3d(pt2)
        enpt = pt2.encrypt(encrypted_calculator)
        enrs = enpt.matmul_3d(arr1, multiply='right')

        rng_generator = random_number_generator.RandomNumberGenerator()

        enpt2 = pt4.encrypt(encrypted_calculator)
        random_num = rng_generator.generate_random_number(enpt2.shape)


if __name__ == '__main__':
    unittest.main()
