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

import os
import random
import gmpy2

POWMOD_GMP_SIZE = pow(2, 64)

def powmod(a, b, c):
    """
    return int: (a ** b) % c
    """
    
    if a == 1:
        return 1
    
    if max(a, b, c) < POWMOD_GMP_SIZE:
        return pow(a, b, c)
    
    else:
        return int(gmpy2.powmod(a, b, c))


def invert(a, b):
    """return int: x, where a * x == 1 mod b
    """    
    x = int(gmpy2.invert(a, b))
   
    if x == 0:
        raise ZeroDivisionError('invert(a, b) no inverse exists')
    
    return x
   
   
def getprimeover(n):
    """return a random n-bit prime number
    """     
    r = gmpy2.mpz(random.SystemRandom().getrandbits(n))
    r = gmpy2.bit_set(r, n - 1)
    
    return int(gmpy2.next_prime(r))


def isqrt(n):
    """ return the integer square root of N """
    
    return int(gmpy2.isqrt(n))


