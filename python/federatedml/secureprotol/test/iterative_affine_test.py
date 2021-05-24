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
        self.randomized_key = IterativeAffineCipher.generate_keypair(randomized=True)
        self.deterministic_key = IterativeAffineCipher.generate_keypair(randomized=False)
         
    def tearDown(self):
        unittest.TestCase.tearDown(self)
         
    def test_int_add_randomized(self):
        x_li = np.ones(100) * np.random.randint(100)
        y_li = np.ones(100) * np.random.randint(1000)        
        z_li = np.ones(100) * np.random.rand()
        t_li = range(100)
        
        for i in range(x_li.shape[0]):
            x = x_li[i]
            y = y_li[i]
            z = z_li[i]
            t = t_li[i]
            en_x = self.randomized_key.encrypt(x)
            en_y = self.randomized_key.encrypt(y)
            en_z = self.randomized_key.encrypt(z)
            en_t = self.randomized_key.encrypt(t)
            
            en_res = en_x + en_y + en_z + en_t
            
            res = x + y + z + t
            
            de_en_res = self.randomized_key.decrypt(en_res)
            self.assertAlmostEqual(de_en_res, res)

    def test_float_add_randomized(self):
        x_li = np.ones(100) * np.random.uniform(-1e6, 1e6)
        y_li = np.ones(100) * np.random.randint(-1e6, 1e6)
        z_li = np.ones(100) * np.random.uniform(-1e6, )
        t_li = range(100)

        for i in range(x_li.shape[0]):
            x = x_li[i]
            y = y_li[i]
            z = z_li[i]
            t = t_li[i]
            en_x = self.randomized_key.encrypt(x)
            en_y = self.randomized_key.encrypt(y)
            en_z = self.randomized_key.encrypt(z)
            en_t = self.randomized_key.encrypt(t)

            en_res = en_x + en_y + en_z + en_t

            res = x + y + z + t

            de_en_res = self.randomized_key.decrypt(en_res)
            self.assertAlmostEqual(de_en_res, res)

    def test_add_randomized(self):
        x_li = np.ones(100) * np.random.randint(100)
        y_li = np.ones(100) * np.random.randint(1000)
        z_li = np.ones(100) * np.random.rand()
        t_li = range(100)

        for i in range(x_li.shape[0]):
            x = x_li[i]
            y = y_li[i]
            z = z_li[i]
            t = t_li[i]
            en_x = self.deterministic_key.encrypt(x)
            en_y = self.deterministic_key.encrypt(y)
            en_z = self.deterministic_key.encrypt(z)
            en_t = self.deterministic_key.encrypt(t)

            en_res = en_x + en_y + en_z + en_t

            res = x + y + z + t

            de_en_res = self.deterministic_key.decrypt(en_res)
            self.assertAlmostEqual(de_en_res, res)

    def test_mul_randomized(self):
        x_li = (np.ones(100) * np.random.randint(100)).tolist()
        y_li = (np.ones(100) * np.random.randint(1000) * -1).tolist()
        z_li = (np.ones(100) * np.random.rand()).tolist()
        t_li = range(100)

        for i in range(len(x_li)):
            x = x_li[i]
            y = int(y_li[i])
            z = z_li[i]
            t = int(t_li[i])
            en_x = self.randomized_key.encrypt(x)
            en_z = self.randomized_key.encrypt(z)

            en_res = (en_x * y + en_z) * t

            res = (x * y + z) * t

            de_en_res = self.randomized_key.decrypt(en_res)
            self.assertAlmostEqual(de_en_res, res)

    def test_int_mul_deterministic(self):
        x_li = (np.ones(100) * np.random.randint(100)).tolist()
        y_li = (np.ones(100) * np.random.randint(1000) * -1).tolist()
        z_li = (np.ones(100) * np.random.rand()).tolist()
        t_li = range(100)

        for i in range(len(x_li)):
            x = x_li[i]
            y = int(y_li[i])
            z = z_li[i]
            t = int(t_li[i])
            en_x = self.deterministic_key.encrypt(x)
            en_z = self.deterministic_key.encrypt(z)

            en_res = (en_x * y + en_z) * t

            res = (x * y + z) * t

            de_en_res = self.deterministic_key.decrypt(en_res)
            self.assertAlmostEqual(de_en_res, res)

    def test_float_mul_deterministic(self):
        N = 100
        x_li = np.random.uniform(-1e5, 1e5, (N, )).tolist()
        y_li = np.random.uniform(-1e5, 1e5, (N, )).tolist()
        en_x = [self.deterministic_key.encrypt(x) for x in x_li]

        for i in range(N):
            xy = x_li[i] * y_li[i]
            en_xy = en_x[i] * y_li[i]

            de_en_res = self.deterministic_key.decrypt(en_xy)
            self.assertAlmostEqual(de_en_res, xy)

    def test_float_add_mul_deterministic(self):
        N = 100
        x_li = np.random.uniform(-1e3, 1e3, (N, )).tolist()
        y_li = np.random.uniform(-1e3, 1e3, (N, )).tolist()
        en_x = [self.deterministic_key.encrypt(x) for x in x_li]

        res = 0
        en_res = en_x[0]
        for i in range(N):
            res += x_li[i] * y_li[i]
            en_res += en_x[i] * y_li[i]

        en_res -= en_x[0]
        de_en_res = self.deterministic_key.decrypt(en_res)
        self.assertAlmostEqual(de_en_res, res)


if __name__ == '__main__': 
    unittest.main()
