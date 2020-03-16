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
class HeteroDecisionTreeTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.dispatch_node_host = self._create_variable(name='dispatch_node_host', src=['guest'], dst=['host'])
        self.dispatch_node_host_result = self._create_variable(name='dispatch_node_host_result', src=['host'], dst=['guest'])
        self.encrypted_grad_and_hess = self._create_variable(name='encrypted_grad_and_hess', src=['guest'], dst=['host'])
        self.encrypted_splitinfo_host = self._create_variable(name='encrypted_splitinfo_host', src=['host'], dst=['guest'])
        self.federated_best_splitinfo_host = self._create_variable(name='federated_best_splitinfo_host', src=['guest'], dst=['host'])
        self.final_splitinfo_host = self._create_variable(name='final_splitinfo_host', src=['host'], dst=['guest'])
        self.node_positions = self._create_variable(name='node_positions', src=['guest'], dst=['host'])
        self.predict_data = self._create_variable(name='predict_data', src=['guest'], dst=['host'])
        self.predict_data_by_host = self._create_variable(name='predict_data_by_host', src=['host'], dst=['guest'])
        self.predict_finish_tag = self._create_variable(name='predict_finish_tag', src=['guest'], dst=['host'])
        self.tree = self._create_variable(name='tree', src=['guest'], dst=['host'])
        self.tree_node_queue = self._create_variable(name='tree_node_queue', src=['guest'], dst=['host'])
