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
class HomoSecureBoostingTreeTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.tree_dim = self._create_variable(name='tree_dim', src=['guest', 'host'], dst=['arbiter'])
        self.feature_number = self._create_variable(name='feature_number', src=['guest', 'host'], dst=['arbiter'])
        self.loss_status = self._create_variable(name='loss_status', src=['guest', 'host'], dst=['arbiter'])
        self.stop_flag = self._create_variable(name='stop_flag', src=['arbiter'], dst=['guest', 'host'])
        self.local_labels = self._create_variable(name='local_labels', src=['guest', 'host'], dst=['arbiter'])
        self.label_mapping = self._create_variable(name='label_mapping', src=['arbiter'], dst=['guest', 'host'])
