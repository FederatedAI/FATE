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
from federatedml.secureprotol.gmpy_math import invert, is_prime, powmod, tonelli, legendre
from federatedml.secureprotol.number_theory.field.base_galois_field import GaloisFieldElement, GaloisFieldArithmetic


class IntegersModuloPrimeElement(GaloisFieldElement):
    """
    A realization of GF: integers modulo a prime
    """

    def __init__(self, val, arithmetic=None):
        """
        :param val: int
        :param arithmetic: IntegersModuloPrimeArithmetic
        """
        super(IntegersModuloPrimeElement, self).__init__()
        if arithmetic is not None:
            # might need rectification
            self.val = arithmetic.rectify(val)
        else:
            # need no rectification
            self.val = val


class IntegersModuloPrimeArithmetic(GaloisFieldArithmetic):
    """
    For the finite field - integers modulo a prime
    """

    def __init__(self, mod):
        add_identity = IntegersModuloPrimeElement(0)
        mul_identity = IntegersModuloPrimeElement(1)
        super(IntegersModuloPrimeArithmetic, self).__init__(add_identity, mul_identity)
        self.mod = mod  # mod base
        self._check_mod_prime()

    def rectify(self, a):
        """
        Rectify an out-of-range element back to this field
        :param a: int
        :return: int
        """
        return a % self.mod

    def add(self, a, b):
        """

        :param a: IntegersModuloPrimeElement
        :param b: IntegersModuloPrimeElement
        :return: IntegersModuloPrimeElement
        """
        if not isinstance(a, IntegersModuloPrimeElement) or not isinstance(b, IntegersModuloPrimeElement):
            raise TypeError("Addition only supports IntegersModuloPrimeElement objects")
        return IntegersModuloPrimeElement((a.val + b.val) % self.mod)

    def neg(self, a):
        """

        :param a: IntegersModuloPrimeElement
        :return: IntegersModuloPrimeElement
        """
        if not isinstance(a, IntegersModuloPrimeElement):
            raise TypeError("Negative only supports IntegersModuloPrimeElement objects")
        return IntegersModuloPrimeElement(self.mod - a.val)

    def sub(self, a, b):
        """

        :param a: IntegersModuloPrimeElement
        :param b: IntegersModuloPrimeElement
        :return: IntegersModuloPrimeElement
        """
        return self.add(a, self.neg(b))

    def mul(self, a, b):
        """

        :param a: IntegersModuloPrimeElement
        :param b: IntegersModuloPrimeElement
        :return: IntegersModuloPrimeElement
        """
        if isinstance(a, IntegersModuloPrimeElement) and isinstance(b, IntegersModuloPrimeElement):
            return IntegersModuloPrimeElement((a.val * b.val) % self.mod)
        elif isinstance(a, IntegersModuloPrimeElement) and isinstance(b, int):
            if b == 0:
                return self.add_identity
            elif b < 0:
                raise ValueError("Scalars in multiplication must be non-negative")
            else:
                return IntegersModuloPrimeElement((a.val * b) % self.mod)
        elif isinstance(a, int) and isinstance(b, IntegersModuloPrimeElement):
            if a == 0:
                return self.add_identity
            elif a < 0:
                raise ValueError("Scalars in multiplication must be non-negative")
            else:
                return IntegersModuloPrimeElement((a * b.val) % self.mod)
        else:
            raise TypeError("Multiplication only supports two IntegersModuloPrimeElement objects" +
                            "one int plus one object")

    def invert(self, a):
        """

        :param a: IntegersModuloPrimeElement
        :return: IntegersModuloPrimeElement
        """
        if not isinstance(a, IntegersModuloPrimeElement):
            raise TypeError("Invert only supports IntegersModuloPrimeElement objects")
        return IntegersModuloPrimeElement(invert(a.val, self.mod))

    def div(self, a, b):
        """

        :param a: IntegersModuloPrimeElement
        :return: IntegersModuloPrimeElement
        """
        if not isinstance(a, IntegersModuloPrimeElement) or not isinstance(b, IntegersModuloPrimeElement):
            raise TypeError("Division only supports IntegersModuloPrimeElement objects")
        return self.mul(a, self.invert(b))

    def pow(self, a, e):
        """

        :param a: IntegersModuloPrimeElement
        :param e: int
        :return: IntegersModuloPrimeElement
        """
        if not isinstance(a, IntegersModuloPrimeElement) or not isinstance(e, int):
            raise TypeError("Power only supports IntegersModuloPrimeElement to the int's")
        if e == 0:
            return self.mul_identity
        elif e < 0:
            raise ValueError("Exponents in power must be non-negative")
        else:
            return IntegersModuloPrimeElement(powmod(a.val, e, self.mod))

    def sqrt(self, a):
        """
        sqrt(a) found by the Tonelli–Shanks algorithm
        :param a: IntegersModuloPrimeElement
        :return: Output -1 if a is not a quadratic residue, otherwise the correct square roots (root, -root)
                Note root < self.mod / 2
        """
        if not isinstance(a, IntegersModuloPrimeElement):
            raise TypeError("Square root only supports an object")
        if self.is_a_quadratic_residue(a):
            root_raw = tonelli(a.val, self.mod)
            root_raw_other = self.mod - root_raw
            if root_raw < root_raw_other:
                return IntegersModuloPrimeElement(root_raw), IntegersModuloPrimeElement(root_raw_other)
            else:
                return IntegersModuloPrimeElement(root_raw_other), IntegersModuloPrimeElement(root_raw)
        else:
            return -1, -1

    def is_a_quadratic_residue(self, a):
        """
        Check if a is a quadratic residue
        :param a: IntegersModuloPrimeElement
        :return:
        """
        if not isinstance(a, IntegersModuloPrimeElement):
            raise ValueError("Only check an object")
        return legendre(a.val, self.mod) == 1

    def is_positive(self, a):
        """
        Check if a is positive in this field, i.e., if a < self.mod / 2
        :param a: IntegersModuloPrimeElement
        :return:
        """
        return a.val < self.mod / 2

    def _check_mod_prime(self):
        if not is_prime(self.mod):
            raise ValueError("Galois fields take only prime orders")

    def get_add_identity(self):
        return self.add_identity

    def get_mul_identity(self):
        return self.mul_identity
