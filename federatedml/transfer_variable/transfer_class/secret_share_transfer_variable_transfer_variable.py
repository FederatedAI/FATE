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
class SecretShareTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.multiply_triplets_cross = self._create_variable(name='multiply_triplets_cross', src=['guest', 'host'], dst=['guest', 'host'])
        self.multiply_triplets_encrypted = self._create_variable(name='multiply_triplets_encrypted', src=['guest', 'host'], dst=['guest', 'host'])
        self.rescontruct = self._create_variable(name='rescontruct', src=['guest', 'host'], dst=['guest', 'host'])
        self.share = self._create_variable(name='share', src=['guest', 'host'], dst=['guest', 'host'])
