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

import math
import numpy as np
import sys
import random
from numba import jit

from federatedml.secureprotol import gmpy_math


class FixedPointNumber(object):
    """Represents a float or int fixedpoit encoding;.
    """
    BASE = 16    
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig
    
    Q = 293973345475167247070445277780365744413
    
    def __init__(self, encoding, exponent, n=None, max_int=None):
        self.n = n
        self.max_int = max_int
        
        if self.n is None:
            self.n = self.Q
            self.max_int = self.Q // 3 - 1
            
        self.encoding = encoding
        self.exponent = exponent
   
    @classmethod
    def encode(cls, scalar, n=None, max_int=None, precision=None, max_exponent=None):
        """return an encoding of an int or float.
        """
        # Calculate the maximum exponent for desired precision
        exponent = None
        
        #  Too low value preprocess;
        #  avoid "OverflowError: int too large to convert to float"
        
        if np.abs(scalar) < 1e-200:
            scalar = 0
        
        if n is None:
            n = cls.Q
            max_int = cls.Q // 3 - 1    
                
        if precision is None:
            if isinstance(scalar, int) or isinstance(scalar, np.int16) or \
              isinstance(scalar, np.int32) or isinstance(scalar, np.int64):
                exponent = 0                                
            elif isinstance(scalar, float) or isinstance(scalar, np.float16)  \
                 or isinstance(scalar, np.float32) or isinstance(scalar, np.float64):               
                flt_exponent = math.frexp(scalar)[1]
                lsb_exponent = cls.FLOAT_MANTISSA_BITS - flt_exponent
                exponent = math.floor(lsb_exponent / cls.LOG2_BASE)
            else:
                raise TypeError("Don't know the precision of type %s."
                                % type(scalar))
        else:
            exponent = math.floor(math.log(precision, cls.BASE))        
        
        if max_exponent is not None:
            exponent = max(max_exponent, exponent)
            
        int_fixpoint = int(round(scalar * pow(cls.BASE, exponent)))

        if abs(int_fixpoint) > max_int:
            raise ValueError('Integer needs to be within +/- %d but got %d'
                             % (max_int, int_fixpoint))

        return cls(int_fixpoint % n, exponent, n, max_int)

    def decode(self):
        """return decode plaintext.
        """
        if self.encoding >= self.n:
            # Should be mod n
            raise ValueError('Attempted to decode corrupted number')
        elif self.encoding <= self.max_int:
            # Positive
            mantissa = self.encoding
        elif self.encoding >= self.n - self.max_int:
            # Negative
            mantissa = self.encoding - self.n
        else:
            raise OverflowError('Overflow detected in decode number')

        return mantissa * pow(self.BASE, -self.exponent)
 
    def increase_exponent_to(self, new_exponent):
        """return FixedPointNumber: new encoding with same value but having great exponent.
        """
        if new_exponent < self.exponent:
            raise ValueError('New exponent %i should be greater than'
                             'old exponent %i' % (new_exponent, self.exponent))
            
        factor = pow(self.BASE, new_exponent - self.exponent)
        new_encoding = self.encoding * factor % self.n
        
        return FixedPointNumber(new_encoding, new_exponent, self.n, self.max_int)

    def __align_exponent(self, x, y):
        """return x,y with same exponet
        """
        if x.exponent < y.exponent:
            x = x.increase_exponent_to(y.exponent)
        elif x.exponent > y.exponent:
            y = y.increase_exponent_to(x.exponent)
        
        return x, y

    def __truncate(self, a):
        scalar = a.decode()
        return FixedPointNumber.encode(scalar)    

    def __add__(self, other):       
        if isinstance(other, FixedPointNumber):
            return self.__add_fixpointnumber(other)
        else:
            return self.__add_scalar(other)    
    
    def __radd__(self, other):        
        return self.__add__(other)
    

    def __sub__(self, other):
        if isinstance(other, FixedPointNumber):
            return self.__sub_fixpointnumber(other)
        else:
            return self.__sub_scalar(other)    
 
    def __rsub__(self, other):
        x = self.__sub__(other)
        x = -1 * x.decode()
        return self.encode(x)
    
    def __rmul__(self, other):
        return self.__mul__(other)
   
    def __mul__(self, other):
        if isinstance(other, FixedPointNumber):
            return self.__mul_fixpointnumber(other)
        else:
            return self.__mul_scalar(other)
 
    def __truediv__(self, other):
        if isinstance(other, FixedPointNumber):
            scalar = other.decode()
        else:
            scalar = other
        
        return self.__mul__(1 / scalar) 
   
    def __rtruediv__(self, other):
        res = 1.0 / self.__truediv__(other).decode() 
        return FixedPointNumber.encode(res)
 
    def __lt__(self, other):
        x = self.decode()
        if isinstance(other, FixedPointNumber):           
            y = other.decode()                      
        else:
            y = other
        if x < y:
            return True
        else:
            return False  

    def __gt__(self, other):
        x = self.decode()
        if isinstance(other, FixedPointNumber):           
            y = other.decode()                      
        else:
            y = other
        if x > y:
            return True
        else:
            return False
   
    def __le__(self, other):
        x = self.decode()
        if isinstance(other, FixedPointNumber):           
            y = other.decode()                      
        else:
            y = other
        if x <= y:
            return True
        else:
            return False  
 
    def __ge__(self, other):
        x = self.decode()
        if isinstance(other, FixedPointNumber):           
            y = other.decode()                      
        else:
            y = other
            
        if x >= y:
            return True
        else:
            return False 
 
    def __eq__(self, other):
        x = self.decode()
        if isinstance(other, FixedPointNumber):           
            y = other.decode()                      
        else:
            y = other
        if x == y:
            return True
        else:
            return False 
   
    def __ne__(self, other):
        x  = self.decode()
        if isinstance(other, FixedPointNumber):           
            y  = other.decode()                      
        else:
            y = other
        if x != y:
            return True
        else:
            return False   
  
    def __add_fixpointnumber(self, other):        
        x, y = self.__align_exponent(self, other)
        encoding = (x.encoding + y.encoding) % self.Q        
        return FixedPointNumber(encoding, x.exponent)
 
    def __add_scalar(self, scalar):
        encoded = self.encode(scalar)
        return self.__add_fixpointnumber(encoded)
  
    def __sub_fixpointnumber(self, other):        
        scalar = -1 * other.decode()
        return self.__add_scalar(scalar)
    
    def __sub_scalar(self, scalar):
        scalar = -1 * scalar
        return self.__add_scalar(scalar)    
  
    def __mul_fixpointnumber(self, other):     
        encoding = (self.encoding * other.encoding) % self.Q
        exponet = self.exponent + other.exponent
        mul_fixedpoint = FixedPointNumber(encoding, exponet)            
        truncate_mul_fixedpoint= self.__truncate(mul_fixedpoint)        
        return truncate_mul_fixedpoint
   
    def __mul_scalar(self, scalar):
        encoded = self.encode(scalar)
        return self.__mul_fixpointnumber(encoded)
