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
from arch.api.utils import log_utils
from federatedml.param.base_param import BaseParam
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class EncryptParam(BaseParam):
    """
    Define encryption method that used in federated ml.

    Parameters
    ----------
    method : str, default: 'Paillier'
        If method is 'Paillier', Paillier encryption will be used for federated ml.
        To use non-encryption version in HomoLR, just set this parameter to be any other str.
        For detail of Paillier encryption, please check out the paper mentioned in README file.

    key_length : int, default: 1024
        Used to specify the length of key in this encryption method. Only needed when method is 'Paillier'

    """

    def __init__(self, method=consts.PAILLIER, key_length=1024):
        super(EncryptParam, self).__init__()
        self.method = method
        self.key_length = key_length

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
            elif user_input == "iterativeaffine":
                self.method = consts.ITERATIVEAFFINE
            else:
                raise ValueError(
                    "encrypt_param's method {} not supported".format(user_input))

        if type(self.key_length).__name__ != "int":
            raise ValueError(
                "encrypt_param's key_length {} not supported, should be int type".format(self.key_length))
        elif self.key_length <= 0:
            raise ValueError(
                "encrypt_param's key_length must be greater or equal to 1")

        LOGGER.debug("Finish encrypt parameter check!")
        return True
