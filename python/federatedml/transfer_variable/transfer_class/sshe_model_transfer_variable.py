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
class SSHEModelTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.share_matrix = self._create_variable(name='share_matrix', src=['guest', "host"], dst=['host', "guest"])
        self.encrypted_share_matrix = self._create_variable(name='encrypted_share_matrix', src=['guest', "host"],
                                                            dst=['host', "guest"])
        self.share_error = self._create_variable(name='share_error', src=["host"], dst=["guest"])
        self.host_prob = self._create_variable(name='host_prob', src=['host'], dst=['guest'])
        self.pubkey = self._create_variable(name='pubkey', src=['guest', "host"], dst=['host', "guest"])
        self.encrypted_host_weights = self._create_variable(name='encrypted_host_weights', src=['guest'], dst=['host'])
        self.loss = self._create_variable(name='loss', src=['host'], dst=['guest'])
        self.is_converged = self._create_variable(name='is_converged', src=['guest'], dst=['host'])
        self.wxy_sum = self._create_variable(name='wxy_sum', src=['guest'], dst=['host'])
