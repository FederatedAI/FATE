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
import random

from federatedml.secureprotol.number_theory.group.twisted_edwards_curve_group import TwistedEdwardsCurveArithmetic
from federatedml.secureprotol.oblivious_transfer.base_oblivious_transfer import ObliviousTransfer
from federatedml.secureprotol.random_oracle.hash_function.sha256 import Sha256
from federatedml.secureprotol.random_oracle.message_authentication_code.sha256_mac import Sha256MAC
from federatedml.transfer_variable.transfer_class.oblivious_transfer_transfer_variable \
    import ObliviousTransferTransferVariable


class HauckObliviousTransfer(ObliviousTransfer):
    """
    An implementation of the work in
        Hauck Eduard, and Julian Loss. "Efficient and universally composable protocols for oblivious transfer
        from the CDH assumption."  2017
    Currently supports only 1-N scenarios
    """

    def __init__(self):
        super(HauckObliviousTransfer, self).__init__()
        self.tec_arithmetic = TwistedEdwardsCurveArithmetic()
        self.hash = Sha256()
        self.mac = None     # the MAC's init needs a key
        self.transfer_variable = ObliviousTransferTransferVariable()

    def _gen_random_scalar(self):
        """
        Generate a random integer over [0, q - 1], where q is the order of the Galois field used
        :return:
        """
        return random.randint(0, self.tec_arithmetic.get_field_order() - 1)

    def _hash_tec_element(self, element):
        """
        Hash a Twisted Edwards Curve element
        :param element: TwistedEdwardsCurveElement
        :return: -1 if hash fails, otherwise the correct TwistedEdwardsCurveElement
        """
        element_bytes = self.tec_arithmetic.encode(element)
        element_digest = self.hash.digest(element_bytes)
        return self.tec_arithmetic.decode(element_digest)

    def _init_mac(self, s, r):
        """
        Init the MAC with key = (S, R)
        :param s, r
        :return:
        """
        key = self.tec_arithmetic.encode(s) + self.tec_arithmetic.encode(r)
        self.mac = Sha256MAC(key)

    def _mac_tec_element(self, element, decode_output=False):
        """
        MAC a Twisted Edwards Curve element
        If decode_output = True, decode the 256-bit bytes to a TEC element, otherwise output 32byte bytes
        :param element: TwistedEdwardsCurveElement
        :return: -1 or the correct TwistedEdwardsCurveElement if decode_output = True, otherwise 32-byte bytes
        """
        element_bytes = self.tec_arithmetic.encode(element)
        if self.mac is None:
            raise ValueError("MAC not initialized")
        element_digest = self.mac.digest(element_bytes)
        if decode_output:
            return self.tec_arithmetic.decode(element_digest)
        else:
            return element_digest
