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
from federatedml.secureprotol.fate_ckks import CKKSEncryptedVector


def assert_small_rel_diff_scalar(first, second, threshold=1, max_percent_diff=1e-5):
    """
    Assert two values have small relative differences
    It compares the percentage difference if absolute error is larger than threshold
    """
    abs_err = abs(first - second)
    if abs_err > threshold:
        percentage_diff = abs((first - second) / (abs(first) + abs(second)))
        if not percentage_diff < max_percent_diff:
            raise AssertionError(
                f"Large percentage error {percentage_diff} > {max_percent_diff}, first:{first}, second: {second}")


def assert_almost_equal_vector(expected, actual):
    mean_squared_error = np.square(expected - actual).sum()
    assert mean_squared_error < 1e-3


def serialize_and_deserialize(obj):
    serialized_obj = pickle.dumps(obj)
    deserialized_obj = pickle.loads(serialized_obj)
    return deserialized_obj


class TestNetworkInteraction(unittest.TestCase):
    def setUp(self):
        self.arbiter_public_key, self.arbiter_private_key = CKKSKeypair.generate_keypair()
        self.public_key_in_client = serialize_and_deserialize(
            self.arbiter_public_key)

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def test_cross_parties_enc_mul(self):
        # Select the value scale
        scale = 10
        # Repeat the process 100 times
        n = 100
        for _ in range(n):
            # Party A initialize value
            v_A = np.random.rand() * scale

            # Party B initialize value
            v_B = np.random.rand() * scale

            # Party B receives encrypted values from Party A
            encrypted_v_A = self.public_key_in_client.encrypt(v_A)
            encrypted_v_from_A = serialize_and_deserialize(encrypted_v_A)

            # Party B do multiplication with its value
            encrypted_result = v_B * encrypted_v_from_A

            # Decrypt the value and validate
            decrypted_result = self.arbiter_private_key.decrypt(
                encrypted_result)
            true_value = v_B * v_A
            assert_small_rel_diff_scalar(decrypted_result, true_value)


class TestCKKSEncryptedVector(unittest.TestCase):
    def setUp(self):
        self.public_key, self.private_key = CKKSKeypair.generate_keypair()
        self.public_key = serialize_and_deserialize(self.public_key)

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def test_add_scalar(self):
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
            assert_small_rel_diff_scalar(de_en_res, res)

    def test_mul_scalar(self):
        x_li = np.ones(100) * np.random.randint(10)
        y_li = np.ones(100) * np.random.randint(10) * -1
        z_li = np.ones(100) * np.random.rand()
        t_li = np.ones(100)

        for i in range(x_li.shape[0]):
            x = x_li[i]
            y = y_li[i]
            z = z_li[i]
            t = t_li[i]
            en_x = self.public_key.encrypt(x)

            en_res = en_x * y

            res = x * y

            de_en_res = self.private_key.decrypt(en_res)
            assert_small_rel_diff_scalar(de_en_res, res)

    def test_enc_mul_scalar(self):
        x_li = np.ones(100) * np.random.randint(10)
        y_li = np.ones(100) * np.random.randint(10) * -1
        z_li = np.ones(100) * np.random.rand()
        t_li = np.ones(100)

        for i in range(x_li.shape[0]):
            x = x_li[i]
            y = y_li[i]
            z = z_li[i]
            t = t_li[i]
            en_x = self.public_key.encrypt(x)
            en_y = self.public_key.encrypt(y)
            en_z = self.public_key.encrypt(z)
            en_t = self.public_key.encrypt(t)

            en_res = en_x * en_y

            res = x * y

            de_en_res = self.private_key.decrypt(en_res)
            assert_small_rel_diff_scalar(de_en_res, res)

    def test_add_vec(self):
        x_vec = np.random.rand(100)
        y_vec = np.random.rand(100)

        enc_x_vec = self.public_key.encrypt(x_vec)
        enc_y_vec = self.public_key.encrypt(y_vec)

        expected = x_vec + y_vec
        actual = self.private_key.decrypt(enc_x_vec + enc_y_vec)

        assert_almost_equal_vector(expected, actual)

    def test_enc_vec_mul_plain_scalar(self):
        x_vec = np.random.rand(100)
        y = np.random.randint(10)

        enc_x_vec = self.public_key.encrypt(x_vec)

        expected = x_vec * y
        actual = self.private_key.decrypt(enc_x_vec * y)

        assert_almost_equal_vector(expected, actual)

    def test_enc_vec_mul_enc_vec(self):
        x_vec = np.random.rand(100)
        y_vec = np.random.rand(100)

        enc_x_vec = self.public_key.encrypt(x_vec)
        enc_y_vec = self.public_key.encrypt(y_vec)

        expected = x_vec * y_vec
        actual = self.private_key.decrypt(enc_x_vec * enc_y_vec)

        assert_almost_equal_vector(expected, actual)


class TestCKKSSerialization(unittest.TestCase):

    def setUp(self):
        self.public_key, self.private_key = CKKSKeypair.generate_keypair()

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def test_public_key_serialization(self):
        public_key, _ = CKKSKeypair.generate_keypair()

        # No error in serialization/deserialization
        try:
            serialize_and_deserialize(public_key)
        except Exception:
            self.fail('Failed to serialize/deserialize public key')

    def test_encrypted_number_serialization(self):
        public_key, private_key = CKKSKeypair.generate_keypair()

        x = 100.0
        encrypted_x = public_key.encrypt(x)

        # No error in serialization/deserialization
        try:
            serialize_and_deserialize(encrypted_x)
        except Exception:
            self.fail('Failed to serialize/deserialize an encrypted number')


if __name__ == '__main__':
    unittest.main()
