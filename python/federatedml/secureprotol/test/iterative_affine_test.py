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
import unittest

from federatedml.secureprotol.iterative_affine import IterativeAffineCipher


class TestAffine(unittest.TestCase):
    def setUp(self):
        self.random_key = IterativeAffineCipher.generate_keypair(randomized=True, key_size=2048)
        self.determine_key = IterativeAffineCipher.generate_keypair(randomized=True, key_size=2048)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
         
    def add_test(self, key):
        x_li = np.ones(100) * np.random.randint(100)
        y_li = np.ones(100) * np.random.randint(1000)        
        z_li = np.ones(100) * np.random.rand()
        t_li = range(100)

        for i in range(x_li.shape[0]):
            x = x_li[i]
            y = y_li[i]
            z = z_li[i]
            t = t_li[i]
            en_x = key.encrypt(x)
            en_y = key.encrypt(y)
            en_z = key.encrypt(z)
            en_t = key.encrypt(t)
            
            en_res = en_x + en_y + en_z + en_t
            
            res = x + y + z + t
            
            de_en_res = key.decrypt(en_res)
            self.assertAlmostEqual(de_en_res, res)

    def test_add_method(self):
        self.add_test(self.random_key)
        self.add_test(self.determine_key)

    def float_add_test(self, key):
        x_li = np.ones(100) * np.random.uniform(-1e6, 1e6)
        y_li = np.ones(100) * np.random.randint(-1e6, 1e6)
        z_li = np.ones(100) * np.random.uniform(-1e6, )
        t_li = range(100)

        for i in range(x_li.shape[0]):
            x = x_li[i]
            y = y_li[i]
            z = z_li[i]
            t = t_li[i]
            en_x = key.encrypt(x)
            en_y = key.encrypt(y)
            en_z = key.encrypt(z)
            en_t = key.encrypt(t)

            en_res = en_x + en_y + en_z + en_t

            res = x + y + z + t

            de_en_res = key.decrypt(en_res)
            self.assertAlmostEqual(de_en_res, res)

    def test_float_add_method(self):
        self.float_add_test(self.random_key)
        self.float_add_test(self.determine_key)

    def mul_test(self, key):
        x_li = (np.ones(100) * np.random.randint(100)).tolist()
        y_li = (np.ones(100) * np.random.randint(1000) * -1).tolist()
        z_li = (np.ones(100) * np.random.rand()).tolist()
        t_li = range(100)

        for i in range(len(x_li)):
            x = x_li[i]
            y = int(y_li[i])
            z = z_li[i]
            t = int(t_li[i])
            en_x = key.encrypt(x)
            en_z = key.encrypt(z)

            en_res = (en_x * y + en_z) * t

            res = (x * y + z) * t

            de_en_res = key.decrypt(en_res)
            self.assertAlmostEqual(de_en_res, res)

    def test_mul_method(self):
        self.mul_test(self.random_key)
        self.mul_test(self.determine_key)

    def float_mul_test(self, key):
        n = 100
        x_li = np.random.uniform(-1e5, 1e5, (n, )).tolist()
        y_li = np.random.uniform(-1e5, 1e5, (n, )).tolist()
        en_x = [key.encrypt(x) for x in x_li]

        for i in range(n):
            xy = x_li[i] * y_li[i]
            en_xy = en_x[i] * y_li[i]

            de_en_res = key.decrypt(en_xy)
            self.assertAlmostEqual(de_en_res, xy)

    def test_float_mul_method(self):
        self.float_mul_test(self.random_key)
        self.float_mul_test(self.determine_key)

    def float_add_mul_test(self, key):
        n = 100
        x_li = np.random.uniform(-1e3, 1e3, (n, )).tolist()
        y_li = np.random.uniform(-1e3, 1e3, (n, )).tolist()
        en_x = [key.encrypt(x) for x in x_li]

        res = 0
        en_res = en_x[0]
        for i in range(n):
            res += x_li[i] * y_li[i]
            en_res += en_x[i] * y_li[i]

        en_res -= en_x[0]
        de_en_res = key.decrypt(en_res)
        self.assertAlmostEqual(de_en_res, res)

    def test_float_add_mul_method(self):
        self.float_add_mul_test(self.random_key)
        self.float_add_mul_test(self.determine_key)


if __name__ == '__main__':
    unittest.main()
