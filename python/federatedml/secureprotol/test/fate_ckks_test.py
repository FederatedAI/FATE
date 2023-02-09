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
import pickle

import numpy as np
import unittest
from federatedml.secureprotol.fate_ckks import CKKSKeypair
from federatedml.secureprotol.fate_ckks import CKKSPublicKey
from federatedml.secureprotol.fate_ckks import CKKSPrivateKey
from federatedml.secureprotol.fate_ckks import CKKSEncryptedNumber


class TestCKKSEncryptedNumber(unittest.TestCase):
    def setUp(self):
        self.public_key, self.private_key = CKKSKeypair.generate_keypair()

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def assert_small_rel_diff(self, first, second, threshold=1, max_percent_diff=1e-5):
        """
        Assert two values have small relative differences
        It compares the percentage difference if absolute error is larger than threshold
        """
        abs_err = abs(first - second)
        if abs_err > threshold:
            percentage_diff = abs((first - second) / (abs(first) + abs(second)))
            if not percentage_diff < max_percent_diff:
                raise AssertionError(f"Large percentage error {percentage_diff} > {max_percent_diff}, first:{first}, second: {second}")

    def test_add(self):
        x_li = np.ones(100) * np.random.randint(100)
        y_li = np.ones(100) * np.random.randint(1000)
        z_li = np.ones(100) * np.random.rand()
        t_li = range(100)

        for i in range(x_li.shape[0]):
            x = x_li[i]
            y = y_li[i]
            z = z_li[i]
            t = t_li[i]

            en_x = self.public_key.encrypt(x)
            en_y = self.public_key.encrypt(y)
            en_z = self.public_key.encrypt(z)
            en_t = self.public_key.encrypt(t)

            en_res = en_x + en_y + en_z + en_t

            res = x + y + z + t

            de_en_res = self.private_key.decrypt(en_res)
            self.assert_small_rel_diff(de_en_res, res)

    def test_mul(self):
        x_li = np.ones(100) * np.random.randint(10)
        y_li = np.ones(100) * np.random.randint(100) * -1
        z_li = np.ones(100) * np.random.rand()
        t_li = range(100)

        for i in range(x_li.shape[0]):
            x = x_li[i]
            y = y_li[i]
            z = z_li[i]
            t = t_li[i]
            en_x = self.public_key.encrypt(x)

            en_res = (en_x * y + z) * t

            res = (x * y + z) * t

            de_en_res = self.private_key.decrypt(en_res)
            self.assert_small_rel_diff(de_en_res, res)

        x = 9
        en_x = self.public_key.encrypt(x)

        for i in range(100):
            en_x = en_x + 5000 - 0.2
            x = x + 5000 - 0.2
            de_en_x = self.private_key.decrypt(en_x)
            self.assert_small_rel_diff(de_en_x, x)

    def test_enc_mul(self):
        x_li = np.ones(100) * np.random.randint(10)
        y_li = np.ones(100) * np.random.randint(100) * -1
        z_li = np.ones(100) * np.random.rand()
        t_li = range(100)

        for i in range(x_li.shape[0]):
            x = x_li[i]
            y = y_li[i]
            z = z_li[i]
            t = t_li[i]
            en_x = self.public_key.encrypt(x)
            en_y = self.public_key.encrypt(y)
            en_z = self.public_key.encrypt(z)
            en_t = self.public_key.encrypt(t)

            en_res = (en_x * en_y + en_z) * en_t

            res = (x * y + z) * t

            de_en_res = self.private_key.decrypt(en_res)
            self.assert_small_rel_diff(de_en_res, res)

        x = 9
        en_x = self.public_key.encrypt(x)

        for i in range(100):
            en_x = en_x + 5000 - 0.2
            x = x + 5000 - 0.2
            de_en_x = self.private_key.decrypt(en_x)
            self.assert_small_rel_diff(de_en_x, x)

    def test_chain_mul(self):
        # Compute 5! in encrypted form
        encrypted_product = self.public_key.encrypt(1.0)
        for value in [1, 2, 3, 4, 5]:
            encrypted_product *= value

        # Check if the result is almost equal to 5! = 120
        self.assertAlmostEqual(self.private_key.decrypt(encrypted_product), 120)

    def test_serialization(self):
        x = 100.0
        encrypted_x = self.public_key.encrypt(x)
        result = self.private_key.decrypt(_serialize_and_deserialize(encrypted_x))
        self.assertAlmostEqual(x, result)


class TestCKKSPublicKeySerialization(unittest.TestCase):
    def test_serialization(self):
        public_key, _ = CKKSKeypair.generate_keypair()
        # Test no error in serialization/deserialization
        _serialize_and_deserialize(public_key)
        # Nothing to assert


def _serialize_and_deserialize(obj):
    serialized_obj = pickle.dumps(obj)
    deserialized_obj = pickle.loads(serialized_obj)
    return deserialized_obj


if __name__ == '__main__':
    unittest.main()
