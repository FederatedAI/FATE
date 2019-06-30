#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from federatedml.util.transfer_variable.base_transfer_variable import BaseTransferVariable, Variable


class HeteroSecureBoostingTreeTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.tree_dim = Variable(name="HeteroSecureBoostingTreeTransferVariable.tree_dim", auth={'src': "guest", 'dst': ['host']})
        self.stop_flag = Variable(name="HeteroSecureBoostingTreeTransferVariable.stop_flag", auth={'src': "guest", 'dst': ['host']})
        pass
