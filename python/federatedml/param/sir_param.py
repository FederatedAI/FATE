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
from federatedml.param.intersect_param import PHParam
from federatedml.util import consts


class SecureInformationRetrievalParam(BaseParam):
    """
    security_level: float [0, 1]; if security_level == 0, then do raw data retrieval
    oblivious_transfer_protocol: OT type, only supports OT_Hauck
    commutative_encryption: the commutative encryption scheme used, only supports CommutativeEncryptionPohligHellman
    non_committing_encryption: the non-committing encryption scheme used, only supports aes
    ph_params: params for Pohlig-Hellman Encryption
    raw_retrieval: bool, perform raw retrieval if raw_retrieval
    """
    def __init__(self, security_level=0.5,
                 oblivious_transfer_protocol=consts.OT_HAUCK,
                 commutative_encryption=consts.CE_PH,
                 non_committing_encryption=consts.AES,
                 ph_params=PHParam(),
                 raw_retrieval=False):
        super(SecureInformationRetrievalParam, self).__init__()
        self.security_level = security_level
        self.oblivious_transfer_protocol = oblivious_transfer_protocol
        self.commutative_encryption = commutative_encryption
        self.non_committing_encryption = non_committing_encryption
        self.ph_params = ph_params
        self.raw_retrieval = raw_retrieval

    def check(self):
        descr = "secure information retrieval param's"
        self.check_decimal_float(self.security_level, descr)
        self.check_and_change_lower(self.oblivious_transfer_protocol, [consts.OT_HAUCK.lower()], descr)
        self.check_and_change_lower(self.commutative_encryption, [consts.CE_PH.lower()], descr)
        self.check_and_change_lower(self.non_committing_encryption, [consts.AES.lower()], descr)
        self.ph_params.check()
        self.check_boolean(self.raw_retrieval, descr)
