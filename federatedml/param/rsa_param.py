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


class RsaParam(BaseParam):
    """
    Define the sample method

    Parameters
    ----------
    rsa_key_n: integer, RSA modulus, default: None
    rsa_key_e: integer, RSA public exponent, default: None
    rsa_key_d: integer, RSA private exponent, default: None
    save_out_table_namespace: str, namespace of dtable where stores the output data. default: None
    save_out_table_name: str, name of dtable where stores the output data. default: None
    """

    def __init__(self, rsa_key_n=None, rsa_key_e=None, rsa_key_d=None, save_out_table_namespace=None, save_out_table_name=None):
        self.rsa_key_n = rsa_key_n
        self.rsa_key_e = rsa_key_e
        self.rsa_key_d = rsa_key_d
        self.save_out_table_namespace = save_out_table_namespace
        self.save_out_table_name = save_out_table_name

    def check(self):
        descr = "rsa param"
        self.check_positive_integer(self.rsa_key_n, descr)
        self.check_positive_integer(self.rsa_key_e, descr)
        self.check_positive_integer(self.rsa_key_d, descr)
        self.check_string(self.save_out_table_namespace, descr)
        self.check_string(self.save_out_table_name, descr)
        return True
