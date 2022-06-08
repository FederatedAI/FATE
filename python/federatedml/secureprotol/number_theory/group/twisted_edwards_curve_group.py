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
from federatedml.secureprotol.number_theory.field.integers_modulo_prime_field import IntegersModuloPrimeArithmetic, \
    IntegersModuloPrimeElement
from federatedml.secureprotol.number_theory.group.cyclc_group import CyclicGroupArithmetic, CyclicGroupElement
from federatedml.util.conversion import int_to_bytes, bytes_to_int, int_to_binary_representation


class TwistedEdwardsCurveElement(CyclicGroupElement):

    def __init__(self, x, y, arithmetic=None):
        super(TwistedEdwardsCurveElement, self).__init__()
        self.x = x  # X-coordinate
        self.y = y  # Y-coordinate
        if arithmetic is not None:
            if not arithmetic.is_in_group(self):
                raise ValueError("This element is not in TEC group")

    def output(self):
        """
        Output (X-coordinate, Y-coordinate)
        :return: str
        """
        return "(" + str(self.x.val) + ", " + str(self.y.val) + ")"


class TwistedEdwardsCurveArithmetic(CyclicGroupArithmetic):
    """
    See Bernstein, Daniel J., et al. "Twisted edwards curves." 2008,
        Bernstein, Daniel J., et al. "High-speed high-security signatures." 2012,
        and https://tools.ietf.org/id/draft-struik-lwig-curve-representations-00.html#dom-parms for more details
    """

    def __init__(self,
                 galois_field_arithmetic=IntegersModuloPrimeArithmetic(2 ** 255 - 19),
                 a=IntegersModuloPrimeElement(2 ** 255 - 20),
                 d=IntegersModuloPrimeElement(
                     37095705934669439343138083508754565189542113879843219016388785533085940283555),
                 identity=TwistedEdwardsCurveElement(IntegersModuloPrimeElement(0), IntegersModuloPrimeElement(1)),
                 generator=None):
        if generator is None:
            super(TwistedEdwardsCurveArithmetic, self).__init__(identity, self.default_generator())
        else:
            super(TwistedEdwardsCurveArithmetic, self).__init__(identity, generator)
        self.FA = galois_field_arithmetic
        self.a = a
        self.d = d

    @staticmethod
    def default_generator():
        x = IntegersModuloPrimeElement(15112221349535400772501151409588531511454012693041857206046113283949847762202)
        y = IntegersModuloPrimeElement(46316835694926478169428394003475163141307993866256225615783033603165251855960)
        return TwistedEdwardsCurveElement(x, y)

    def is_in_group(self, element):
        x = element.x
        y = element.y

        # left = ax^2 + y^2
        ax_square = self.FA.mul(self.a, self.FA.pow(x, 2))
        y_square = self.FA.pow(y, 2)
        left = self.FA.add(ax_square, y_square)

        # right = 1 + dx^2y^2
        one = self.FA.get_mul_identity()
        dx_square_y_square = self.FA.mul(self.d, self.FA.mul(self.FA.pow(x, 2), self.FA.pow(y, 2)))
        right = self.FA.add(one, dx_square_y_square)

        # check if left == right
        if self.FA.sub(left, right) == self.FA.get_add_identity():
            return True
        else:
            return False

    def add(self, a, b):
        """
        (x1, y1) + (x2, y2) = ((x1y2 + y1x2) / (1 + dx1x2y1y2), (y1y2 - ax1x2) / (1 - dx1x2y1y2))
        :param a: TwistedEdwardsCurveElement
        :param b: TwistedEdwardsCurveElement
        :return:
        """
        if not isinstance(a, TwistedEdwardsCurveElement) or not isinstance(b, TwistedEdwardsCurveElement):
            raise TypeError("Addition only supports two objects")
        x1 = a.x
        y1 = a.y
        x2 = b.x
        y2 = b.y

        # calculate essential components
        x1y2 = self.FA.mul(x1, y2)
        x2y1 = self.FA.mul(x2, y1)
        ax1x2 = self.FA.mul(self.a, self.FA.mul(x1, x2))
        y1y2 = self.FA.mul(y1, y2)
        dx1x2y1y2 = self.FA.mul(self.d, self.FA.mul(x1y2, x2y1))

        # calculate the first coordinate
        numerator_x3 = self.FA.add(x1y2, x2y1)
        denominator_x3 = self.FA.add(self.FA.get_mul_identity(), dx1x2y1y2)
        x3 = self.FA.div(numerator_x3, denominator_x3)

        # calculate the second coordinate
        numerator_y3 = self.FA.sub(y1y2, ax1x2)
        denominator_y3 = self.FA.sub(self.FA.get_mul_identity(), dx1x2y1y2)
        y3 = self.FA.div(numerator_y3, denominator_y3)

        return TwistedEdwardsCurveElement(x3, y3)

    def neg(self, a):
        """
        -(x, y) = (-x, y)
        :param a: TwistedEdwardsCurveElement
        :return:
        """
        if not isinstance(a, TwistedEdwardsCurveElement):
            raise TypeError("Negative only supports an object")
        x = a.x
        y = a.y
        return TwistedEdwardsCurveElement(self.FA.neg(x), y)

    def sub(self, a, b):
        """

        :param a: TwistedEdwardsCurveElement
        :param b: TwistedEdwardsCurveElement
        :return:
        """
        return self.add(a, self.neg(b))

    def mul(self, scalar, a):
        """

        :param scalar: int
        :param a: TwistedEdwardsCurveElement
        :return:
        """
        if not isinstance(scalar, int) or not isinstance(a, TwistedEdwardsCurveElement):
            raise TypeError("Multiplication only supports a scalar with an object")
        if scalar == 0:
            return self.get_identity()
        elif scalar < 0:
            raise TypeError("Multiplication only supports non-negative scalars")
        else:
            binary_representation = int_to_binary_representation(scalar)
            res = self.identity
            for exponent in binary_representation:
                res = self.add(res, self._multiple_twice(exponent, a))
            return res

    def _twice(self, a):
        """
        2 * (x, y) = (2xy / (ax^2 + y^2), (y^2 - ax^2) / (2 - ax^2 - y^2))
        :param a: TwistedEdwardsCurveElement
        :return:
        """
        if not isinstance(a, TwistedEdwardsCurveElement):
            raise TypeError("Double only supports an object")
        x = a.x
        y = a.y

        # calculate essential components
        ax_square = self.FA.mul(self.a, self.FA.pow(x, 2))
        y_square = self.FA.pow(y, 2)
        two = self.FA.mul(2, self.FA.get_mul_identity())

        # calculate the first coordinate
        numerator_x3 = self.FA.mul(2, self.FA.mul(x, y))
        denominator_x3 = self.FA.add(ax_square, y_square)
        x3 = self.FA.div(numerator_x3, denominator_x3)

        # calculate the second coordinate
        numerator_y3 = self.FA.sub(y_square, ax_square)
        denominator_y3 = self.FA.sub(two, denominator_x3)
        y3 = self.FA.div(numerator_y3, denominator_y3)

        return TwistedEdwardsCurveElement(x3, y3)

    def _multiple_twice(self, multiple, a):
        """
        2^multiple * a
        :param multiple: int >= 0
        :param a: TwistedEdwardsCurveElement
        :return:
        """
        if multiple == 0:
            return a
        else:
            res = a
            for i in range(multiple):
                res = self._twice(res)
            return res

    def encode(self, a):
        """
        Encode an element to a 33-byte bytes for feeding into a cryptographic hash function
        :param a: TwistedEdwardsCurveElement
        :return:
        """
        pos_sign = "00"
        neg_sign = "FF"
        if self.FA.is_positive(a.x):
            return bytes.fromhex(pos_sign) + int_to_bytes(a.y.val)
        else:
            return bytes.fromhex(neg_sign) + int_to_bytes(a.y.val)

    def decode(self, bytes_arr: bytes):
        """
        Decode a bytes objects, expected to be 32 bytes (256 bits) long, into a self-typed object
        Note that this decode is not simply a reverse of the encode above
        :param bytes_arr:
        :return: Output -1 is the result is not in the TEC group, otherwise the correct answer
        """
        if len(bytes_arr) % 2 != 0:
            raise ValueError("Cannot decode an odd-long bytes into a TEC element")
        y = IntegersModuloPrimeElement(bytes_to_int(bytes_arr), arithmetic=self.FA)

        # determine x
        denominator = self.FA.sub(self.a, self.FA.mul(self.d, self.FA.pow(y, 2)))   # a - dy^2
        numerator = self.FA.sub(self.FA.get_mul_identity(), self.FA.pow(y, 2))
        x_pos, x_neg = self.FA.sqrt(self.FA.div(numerator, denominator))

        # if the decoded object is invalid, return -1
        if isinstance(x_pos, int) and x_pos == -1:
            return -1

        # if the first byte of the byte_arr is less than half a byte's bit-length,
        #   then use the positive square root as x, otherwise negative
        x = x_pos if bytes_arr[0] < 128 else x_neg
        return TwistedEdwardsCurveElement(x, y)

    def get_field_order(self):
        return self.FA.mod
