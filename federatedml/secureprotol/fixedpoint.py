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


class FixedPointNumber(object):
    """Represents a float or int fixedpoit encoding;.
    """
    BASE = 16    
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig

    def __init__(self, n, encoding, exponent):
        self.n = n
        self.max_int = n // 3 - 1
        self.encoding = encoding
        self.exponent = exponent

    @classmethod
    def encode(cls, n, max_int, scalar, precision=None, max_exponent=None):
        """return an encoding of an int or float.
        """
        # Calculate the maximum exponent for desired precision
        exponent = None        
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

        return cls(n, int_fixpoint % n, exponent)

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
        
        return self.__class__(self.n, new_encoding, new_exponent)
    
    