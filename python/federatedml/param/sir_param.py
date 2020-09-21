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
from federatedml.util import consts


class SecureInformationRetrievalParam(BaseParam):
    """
    security_level: float [0, 1]; if security_level == 0, then do raw data retrieval
    oblivious_transfer_protocol: OT type, only supports consts.OT_HAUCK
    commutative_encryption: the commutative encryption scheme used, only supports consts.CE_PH
    non_committing_encryption: the non-committing encryption scheme used, only supports consts.AES
    key_size: int >= 768, the key length of the commutative cipher
    raw_retrieval: bool, perform raw retrieval if raw_retrieval
    """
    def __init__(self, security_level=0.5,
                 oblivious_transfer_protocol=consts.OT_HAUCK,
                 commutative_encryption=consts.CE_PH,
                 non_committing_encryption=consts.AES,
                 key_size=1024,
                 raw_retrieval=False):
        """

        :param security_level: float
            oblivious_transfer_protocol: str
            commutative_encryption: str
            non_committing_encryption: str
            key_size: int
            raw_retrieval: bool
        """
        super(SecureInformationRetrievalParam, self).__init__()
        self.security_level = security_level
        self.oblivious_transfer_protocol = oblivious_transfer_protocol
        self.commutative_encryption = commutative_encryption
        self.non_committing_encryption = non_committing_encryption
        self.key_size = key_size
        self.raw_retrieval = raw_retrieval

    def check(self):
        descr = "secure information retrieval param's"
        self.check_decimal_float(self.security_level, descr)
        self.check_string(self.oblivious_transfer_protocol, descr)
        self.check_string(self.commutative_encryption, descr)
        self.check_string(self.non_committing_encryption, descr)
        self.check_positive_integer(self.key_size, descr)
        self.check_boolean(self.raw_retrieval, descr)
