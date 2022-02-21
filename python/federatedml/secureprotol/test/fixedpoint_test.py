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
from federatedml.secureprotol.fixedpoint import FixedPointNumber

class TestFixedPointNumber(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        
    def tearDown(self):
        unittest.TestCase.tearDown(self)
        
    def test_encode_decode(self):
        for i in range(100):
            en_i = FixedPointNumber.encode(i)
            de_en_i = en_i.decode()
            self.assertEqual(de_en_i, i)
            en_i = FixedPointNumber.encode(-i)
            de_en_i = en_i.decode()
            self.assertEqual(de_en_i, -i)
        
        for i in range(100):
            x = i * 0.6
            en_x = FixedPointNumber.encode(x)
            de_en_x = en_x.decode()
            self.assertAlmostEqual(de_en_x, x)
               
        elem = np.ones(100) * np.random.rand()
        for x in elem:
            en_x = FixedPointNumber.encode(x)
            de_en_x = en_x.decode()
            self.assertAlmostEqual(de_en_x, x)
        
        elem = np.ones(100) * np.random.randint(100)
        for x in elem:
            en_x = FixedPointNumber.encode(x)
            de_en_x = en_x.decode()
            self.assertAlmostEqual(de_en_x, x)
    
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
            en_x = FixedPointNumber.encode(x)
            en_y = FixedPointNumber.encode(y)
            en_z = FixedPointNumber.encode(-z)
            en_t = FixedPointNumber.encode(-t)
            
            en_res = en_x + en_y + en_z + en_t
            
            res = x + y + (-z) + (-t)
            
            de_en_res = en_res.decode()
            self.assertAlmostEqual(de_en_res, res)
    
    def test_sub(self):
        x_li = np.ones(100) * np.random.randint(100)
        y_li = np.ones(100) * np.random.randint(1000)        
        z_li = np.ones(100) * np.random.rand()
        t_li = range(100)
        
        for i in range(x_li.shape[0]):
            x = x_li[i]
            y = y_li[i]
            z = z_li[i]
            t = t_li[i]
            en_x = FixedPointNumber.encode(x)
            en_y = FixedPointNumber.encode(y)
            en_z = FixedPointNumber.encode(z)
            en_t = FixedPointNumber.encode(t)
            
            en_res = en_x - en_y - en_z - en_t
            
            res = x - y - z - t
            
            de_en_res = en_res.decode()
            self.assertAlmostEqual(de_en_res, res)
            
    def test_mul(self):
        x_li = np.ones(100) * np.random.randint(100)
        y_li = np.ones(100) * np.random.randint(1000) * -1        
        z_li = np.ones(100) * np.random.rand()
        t_li = range(0, 100)
        
        for i in range(x_li.shape[0]):
            x = x_li[i]
            y = y_li[i]
            z = z_li[i]
            t = t_li[i]
            en_x = FixedPointNumber.encode(x)
             
            en_res = (en_x * y + z) * t
             
            res = (x * y + z) * t
                                 
            de_en_res = en_res.decode()
            self.assertAlmostEqual(de_en_res, res)
        
        x = 9
        en_x = FixedPointNumber.encode(x)
        for i in range(100):
            en_x = en_x + 5000 - 0.2
            x = x + 5000 - 0.2
            de_en_x = en_x.decode()
            self.assertAlmostEqual(de_en_x, x)
    
    def test_div(self):
        for i in range(100):
            x = np.random.randn() * 100 
            y = np.random.randn() * 100
            en_x = FixedPointNumber.encode(x)
            en_y = FixedPointNumber.encode(y)
            z = x / y
            en_z = en_x / en_y
            de_en_z = en_z.decode()
            self.assertAlmostEqual(de_en_z, z)   
    
    def test_lt(self):
        for i in range(100):
            x = np.random.randn() * 100 
            y = np.random.randn() * 100
            en_x = FixedPointNumber.encode(x)
            en_y = FixedPointNumber.encode(y)
            z = x < y
            en_z = en_x < en_y            
            self.assertEqual(en_z, z)
    
    def test_gt(self):
        for i in range(100):
            x = np.random.randn() * 100 
            y = np.random.randn() * 100
            en_x = FixedPointNumber.encode(x)
            en_y = FixedPointNumber.encode(y)
            z = x > y
            en_z = en_x > en_y            
            self.assertEqual(en_z, z) 
    
    def test_le(self):
        for i in range(100):
            x = np.random.randint(10)
            y = np.random.randint(10)
            en_x = FixedPointNumber.encode(x)
            en_y = FixedPointNumber.encode(y)
            z = x <= y
            en_z = en_x <= en_y            
            self.assertEqual(en_z, z) 
    
    def test_ge(self):
        for i in range(100):
            x = np.random.randint(10)
            y = np.random.randint(10)
            en_x = FixedPointNumber.encode(x)
            en_y = FixedPointNumber.encode(y)
            z = x >= y
            en_z = en_x >= en_y            
            self.assertEqual(en_z, z) 
    
    def test_eq(self):
        for i in range(100):
            x = np.random.randint(10)
            y = np.random.randint(10)
            en_x = FixedPointNumber.encode(x)
            en_y = FixedPointNumber.encode(y)
            z = x == y
            en_z = en_x == en_y            
            self.assertEqual(en_z, z) 
    
    def test_ne(self):
        for i in range(100):
            x = np.random.randint(10)
            y = np.random.randint(10)            
            en_x = FixedPointNumber.encode(x)
            en_y = FixedPointNumber.encode(y)
            z = x != y
            en_z = en_x != en_y
            
            self.assertEqual(en_z, z) 
            
   
if __name__ == '__main__': 
    unittest.main()
    
    
    