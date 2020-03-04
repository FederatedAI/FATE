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
import unittest

import numpy as np

from arch.api.session import init
from federatedml.ftl.eggroll_computation.helper import distribute_encrypt_matrix
from federatedml.ftl.encryption import encryption
from federatedml.secureprotol.encrypt import PaillierEncrypt


class TestEncryptionMatmul(unittest.TestCase):

    def setUp(self):
        paillierEncrypt = PaillierEncrypt()
        paillierEncrypt.generate_key()
        self.publickey = paillierEncrypt.get_public_key()
        self.privatekey = paillierEncrypt.get_privacy_key()

    def test_parallel_sequence_running_time(self):
        return

        X = np.ones((50, 50))

        curr_time1 = time.time()
        encryption.encrypt_matrix(self.publickey, X)
        curr_time2 = time.time()
        seq_running_time = curr_time2 - curr_time1
        distribute_encrypt_matrix(self.publickey, X)
        curr_time3 = time.time()
        parallel_running_time = curr_time3 - curr_time2

        assert seq_running_time - parallel_running_time > 0


if __name__ == '__main__':
    init()
    unittest.main()
