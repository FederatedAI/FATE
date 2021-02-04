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
class FeldmanVerifiableSumTransferVariables(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.guest_share_subkey = self._create_variable(name='guest_share_subkey', src=['guest'], dst=['host'])
        self.host_share_to_guest = self._create_variable(name='host_share_to_guest', src=['host'], dst=['guest'])
        self.host_share_to_host = self._create_variable(name='host_share_to_host', src=['host'], dst=['host'])
        self.host_sum = self._create_variable(name='host_sum', src=['host'], dst=['guest'])
        self.guest_commitments = self._create_variable(name='guest_commitments', src=['guest'], dst=['host'])
        self.host_commitments = self._create_variable(name='host_commitments', src=['host'], dst=['host', 'guest'])
