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

from federatedml.util.transfer_variable.base_transfer_variable import BaseTransferVariable, Variable
from federatedml.util.transfer_variable.base_transfer_variable import Variable


class HeteroDecisionTreeTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.encrypted_grad_and_hess = Variable(name="HeteroDecisionTreeTransferVariable.encrypted_grad_and_hess", auth={'src': "guest", 'dst': ['host']})
        self.tree_node_queue = Variable(name="HeteroDecisionTreeTransferVariable.tree_node_queue", auth={'src': "guest", 'dst': ['host']})
        self.node_positions = Variable(name="HeteroDecisionTreeTransferVariable.node_positions", auth={'src': "guest", 'dst': ['host']})
        self.encrypted_splitinfo_host = Variable(name="HeteroDecisionTreeTransferVariable.encrypted_splitinfo_host", auth={'src': "host", 'dst': ['guest']})
        self.federated_best_splitinfo_host = Variable(name="HeteroDecisionTreeTransferVariable.federated_best_splitinfo_host", auth={'src': "guest", 'dst': ['host']})
        self.final_splitinfo_host = Variable(name="HeteroDecisionTreeTransferVariable.final_splitinfo_host", auth={'src': "host", 'dst': ['guest']})
        self.dispatch_node_host = Variable(name="HeteroDecisionTreeTransferVariable.dispatch_node_host", auth={'src': "guest", 'dst': ['host']})
        self.dispatch_node_host_result = Variable(name="HeteroDecisionTreeTransferVariable.dispatch_node_host_result", auth={'src': "host", 'dst': ['guest']})
        self.tree = Variable(name="HeteroDecisionTreeTransferVariable.tree", auth={'src': "guest", 'dst': ['host']})
        self.predict_data = Variable(name="HeteroDecisionTreeTransferVariable.predict_data", auth={'src': "guest", 'dst': ['host']})
        self.predict_data_by_host = Variable(name="HeteroDecisionTreeTransferVariable.predict_data_by_host", auth={'src': "host", 'dst': ['guest']})
        self.predict_finish_tag = Variable(name="HeteroDecisionTreeTransferVariable.predict_finish_tag", auth={'src': "guest", 'dst': ['host']})
        pass
