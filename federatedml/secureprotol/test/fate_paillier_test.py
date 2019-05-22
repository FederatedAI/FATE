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
from federatedml.secureprotol.fate_paillier import PaillierKeypair
from federatedml.secureprotol.fate_paillier import PaillierPublicKey
from federatedml.secureprotol.fate_paillier import PaillierPrivateKey
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber


class TestPaillierEncryptedNumber(unittest.TestCase):
    def setUp(self):
        self.public_key, self.private_key = PaillierKeypair.generate_keypair()
         
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
            en_x = self.public_key.encrypt(x)
            en_y = self.public_key.encrypt(y)
            en_z = self.public_key.encrypt(z)
            en_t = self.public_key.encrypt(t)
            
            en_res = en_x + en_y + en_z + en_t
            
            res = x + y + z + t
            
            de_en_res = self.private_key.decrypt(en_res)
            self.assertAlmostEqual(de_en_res, res)
 
    def test_mul(self):
        x_li = np.ones(100) * np.random.randint(100)
        y_li = np.ones(100) * np.random.randint(1000) * -1        
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
            self.assertAlmostEqual(de_en_res, res)
        
        x = 9
        en_x = self.public_key.encrypt(x)
        
        for i in range(100):
            en_x = en_x + 5000 - 0.2
            x = x + 5000 - 0.2
            de_en_x = self.private_key.decrypt(en_x)
            self.assertAlmostEqual(de_en_x, x)
            
   
if __name__ == '__main__': 
    unittest.main()
    
    
    