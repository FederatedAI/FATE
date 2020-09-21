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
        self.key = IterativeAffineCipher.generate_keypair()
         
    def tearDown(self):
        unittest.TestCase.tearDown(self)
         
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
            en_x = self.key.encrypt(x)
            en_y = self.key.encrypt(y)
            en_z = self.key.encrypt(z)
            en_t = self.key.encrypt(t)
            
            en_res = en_x + en_y + en_z + en_t
            
            res = x + y + z + t
            
            de_en_res = self.key.decrypt(en_res)
            self.assertAlmostEqual(de_en_res, res)
            
   
if __name__ == '__main__': 
    unittest.main()
