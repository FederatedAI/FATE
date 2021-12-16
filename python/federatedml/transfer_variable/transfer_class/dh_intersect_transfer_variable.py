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
class DhIntersectTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.commutative_cipher_public_knowledge = self._create_variable(
            name='commutative_cipher_public_knowledge', src=['guest'], dst=['host'])
        self.id_ciphertext_list_exchange_g2h = self._create_variable(
            name='id_ciphertext_list_exchange_g2h', src=['guest'], dst=['host'])
        self.id_ciphertext_list_exchange_h2g = self._create_variable(
            name='id_ciphertext_list_exchange_h2g', src=['host'], dst=['guest'])
        self.doubly_encrypted_id_list = self._create_variable(
            name='doubly_encrypted_id_list', src=['host'], dst=['guest'])
        self.intersect_ids = self._create_variable(name='intersect_ids', src=['guest'], dst=['host'])
        self.cardinality = self._create_variable(name='cardinality', src=['guest'], dst=['host'])
        self.host_filter = self._create_variable(name='host_filter', src=['host'], dst=['guest'])
