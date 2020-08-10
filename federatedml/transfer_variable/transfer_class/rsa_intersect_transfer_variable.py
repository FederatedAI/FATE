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
class RsaIntersectTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.cache_version_info = self._create_variable(name='cache_version_info', src=['guest'], dst=['host'])
        self.cache_version_match_info = self._create_variable(name='cache_version_match_info', src=['host'], dst=['guest'])
        self.intersect_guest_ids = self._create_variable(name='intersect_guest_ids', src=['guest'], dst=['host'])
        self.intersect_guest_ids_process = self._create_variable(name='intersect_guest_ids_process', src=['host'], dst=['guest'])
        self.intersect_host_ids_process = self._create_variable(name='intersect_host_ids_process', src=['host'], dst=['guest'])
        self.intersect_ids = self._create_variable(name='intersect_ids', src=['guest'], dst=['host'])
        self.rsa_pubkey = self._create_variable(name='rsa_pubkey', src=['host'], dst=['guest'])
