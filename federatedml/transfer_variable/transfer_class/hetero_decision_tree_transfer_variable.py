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

from federatedml.transfer_variable.transfer_class.base_transfer_variable import BaseTransferVariable, Variable


# noinspection PyAttributeOutsideInit
class HeteroDecisionTreeTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.encrypted_grad_and_hess = Variable(name='HeteroDecisionTreeTransferVariable.encrypted_grad_and_hess', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.tree_node_queue = Variable(name='HeteroDecisionTreeTransferVariable.tree_node_queue', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.node_positions = Variable(name='HeteroDecisionTreeTransferVariable.node_positions', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.encrypted_splitinfo_host = Variable(name='HeteroDecisionTreeTransferVariable.encrypted_splitinfo_host', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.federated_best_splitinfo_host = Variable(name='HeteroDecisionTreeTransferVariable.federated_best_splitinfo_host', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.final_splitinfo_host = Variable(name='HeteroDecisionTreeTransferVariable.final_splitinfo_host', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.dispatch_node_host = Variable(name='HeteroDecisionTreeTransferVariable.dispatch_node_host', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.dispatch_node_host_result = Variable(name='HeteroDecisionTreeTransferVariable.dispatch_node_host_result', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.tree = Variable(name='HeteroDecisionTreeTransferVariable.tree', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.predict_data = Variable(name='HeteroDecisionTreeTransferVariable.predict_data', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.predict_data_by_host = Variable(name='HeteroDecisionTreeTransferVariable.predict_data_by_host', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.predict_finish_tag = Variable(name='HeteroDecisionTreeTransferVariable.predict_finish_tag', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        pass
