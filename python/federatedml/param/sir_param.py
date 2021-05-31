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
    target_cols: str or list of str, target cols to retrieve; to retrieve label, please specify "label",
        any values not retrieved will be marked as "unretrieved",
        default None
    target_indexes: int or list of int, target indexes to retrieve, note that label can only be specified by target_cols;
        values will be merged with target_cols,
        default None
    """
    def __init__(self, security_level=0.5,
                 oblivious_transfer_protocol=consts.OT_HAUCK,
                 commutative_encryption=consts.CE_PH,
                 non_committing_encryption=consts.AES,
                 ph_params=PHParam(),
                 raw_retrieval=False,
                 target_cols=None,
                 target_indexes=None):
        super(SecureInformationRetrievalParam, self).__init__()
        self.security_level = security_level
        self.oblivious_transfer_protocol = oblivious_transfer_protocol
        self.commutative_encryption = commutative_encryption
        self.non_committing_encryption = non_committing_encryption
        self.ph_params = ph_params
        self.raw_retrieval = raw_retrieval
        self.target_cols = target_cols
        self.target_indexes = target_indexes

    def check(self):
        descr = "secure information retrieval param's "
        self.check_decimal_float(self.security_level, descr+"security_level")
        self.check_and_change_lower(self.oblivious_transfer_protocol,
                                    [consts.OT_HAUCK.lower()],
                                    descr+"oblivious_transfer_protocol")
        self.check_and_change_lower(self.commutative_encryption,
                                    [consts.CE_PH.lower()],
                                    descr+"commutative_encryption")
        self.check_and_change_lower(self.non_committing_encryption,
                                    [consts.AES.lower()],
                                    descr+"non_committing_encryption")
        self.ph_params.check()
        self.check_boolean(self.raw_retrieval, descr)
        if self.target_cols:
            if not isinstance(self.target_cols, list):
                self.target_cols = [self.target_cols]
            for col in self.target_cols:
                self.check_string(col, descr+"target_cols")
        if self.target_indexes:
            if not isinstance(self.target_indexes, list):
                self.target_indexes = [self.target_indexes]
            for i in self.target_indexes:
                self.check_nonnegative_number(i, descr+"target_indexes")