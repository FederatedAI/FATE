#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


class GaloisFieldElement(object):
    """
    Element of a finite field
    """

    def __init__(self):
        pass

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class GaloisFieldArithmetic(object):
    """
    A collection of arithmetic operators for finite field elements
    """

    def __init__(self, add_identity, mul_identity):
        self.add_identity = add_identity    # additive identity
        self.mul_identity = mul_identity    # multiplicative identity

    def add(self, a, b):
        """
        a + b
        """
        pass

    def neg(self, a):
        """
        -a
        """
        pass

    def sub(self, a, b):
        """
        a - b
        """
        pass

    def mul(self, a, b):
        """
        a * b
        """
        pass

    def invert(self, a):
        """
        a^(-1)
        """
        pass

    def div(self, a, b):
        """
        a / b
        """
        pass

    def pow(self, a, e):
        """
        a^e
        """
        pass

    def get_add_identity(self):
        return self.add_identity

    def get_mul_identity(self):
        return self.mul_identity
