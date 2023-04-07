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
from federatedml.param.base_param import BaseParam
from federatedml.util import consts, LOGGER


class EncryptParam(BaseParam):
    """
    Define encryption method that used in federated ml.

    Parameters
    ----------
    method : {'Paillier'}
        If method is 'Paillier', Paillier encryption will be used for federated ml.
        To use non-encryption version in HomoLR, set this to None.
        For detail of Paillier encryption, please check out the paper mentioned in README file.
    key_length : int, default: 1024
        Used to specify the length of key in this encryption method.

    See https://github.com/microsoft/APSI for details about CKKS's parameters
    """

    def __init__(self, method=consts.PAILLIER, key_length=1024, poly_modulus_degree=None, coeff_mod_bit_sizes=None, global_scale=2 ** 40):
        super(EncryptParam, self).__init__()
        self.method = method
        self.key_length = key_length
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = global_scale

    def check(self):
        if self.method is not None and type(self.method).__name__ != "str":
            raise ValueError(
                "encrypt_param's method {} not supported, should be str type".format(
                    self.method))
        elif self.method is None:
            pass
        else:
            user_input = self.method.lower()
            if user_input == "paillier":
                self.method = consts.PAILLIER
            elif user_input == consts.ITERATIVEAFFINE.lower() or user_input == consts.RANDOM_ITERATIVEAFFINE:
                LOGGER.warning('Iterative Affine and Random Iterative Affine are not supported in version>=1.7.1 '
                               'due to safety concerns, encrypt method will be reset to Paillier')
                self.method = consts.PAILLIER
            elif user_input == "ipcl":
                self.method = consts.PAILLIER_IPCL
            elif user_input == "ckks":
                self.method = consts.CKKS
                LOGGER.warning('CKKS is not fully tested and developed, use it on your own risk')
            else:
                raise ValueError(
                    "encrypt_param's method {} not supported".format(user_input))

        if type(self.key_length).__name__ != "int":
            raise ValueError(
                "encrypt_param's key_length {} not supported, should be int type".format(self.key_length))
        elif self.key_length <= 0:
            raise ValueError(
                "encrypt_param's key_length must be greater or equal to 1")

        if self.poly_modulus_degree is None and self.coeff_mod_bit_sizes is None:
            self.poly_modulus_degree = 8192
            self.coeff_mod_bit_sizes = [60, 40, 40, 60]
            LOGGER.info("No value for poly_modulus_degree and coeff_mod_bit_sizes are set, using default values 8192 and [60, 40, 40, 60]")
        elif self.poly_modulus_degree is not None and self.poly_modulus_degree is not None:
            # Check type
            if type(self.poly_modulus_degree).__name__ != "int":
                raise ValueError(
                    "encrypt_param's poly_modulus_degree {} not supported, should be int type".format(self.poly_modulus_degree))
            if type(self.coeff_mod_bit_sizes).__name__ != "list":
                raise ValueError(
                    "encrypt_param's coeff_mod_bit_sizes {} not supported, should be list type".format(self.coeff_mod_bit_sizes))

            # poly_modulus_degree must be a power of 2
            def is_power_of_two(n):
                return (n & (n - 1) == 0) and n != 0
            if not is_power_of_two(self.poly_modulus_degree):
                raise ValueError("poly_modulus_degree should be a power of two")

            # Sum of coeff_mod_bit_sizes cannot exceed a value given poly_modulus_degree
            valid_bit_sizes_table = {
                1024: 27,
                2048: 54,
                4096: 109,
                8192: 218,
                16384: 438,
                32768: 881
            }
            max_sum = valid_bit_sizes_table[self.poly_modulus_degree]
            if sum(self.coeff_mod_bit_sizes) > max_sum:
                raise ValueError("The sum of coeff_mod_bit_sizes is too large, see https://github.com/microsoft/APSI for details on how to set this")

        else:
            raise ValueError("poly_modulus_degree and coeff_mod_bit_sizes must be either both None or has value")

        LOGGER.debug("Finish encrypt parameter check!")
        return True
