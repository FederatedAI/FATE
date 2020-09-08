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

################################################################################
#
# AUTO GENERATED TRANSFER VARIABLE CLASS. DO NOT MODIFY
#
################################################################################

from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables


# noinspection PyAttributeOutsideInit
class PaillierCipherTransVar(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.use_encrypt = self._create_variable(name='use_encrypt', src=['host'], dst=['arbiter'])
        self.pailler_pubkey = self._create_variable(name='pailler_pubkey', src=['arbiter'], dst=['host'])
        self.re_encrypt_times = self._create_variable(name='re_encrypt_times', src=['host'], dst=['arbiter'])
        self.model_to_re_encrypt = self._create_variable(name='model_to_re_encrypt', src=['host'], dst=['arbiter'])
        self.model_re_encrypted = self._create_variable(name='model_re_encrypted', src=['arbiter'], dst=['host'])
