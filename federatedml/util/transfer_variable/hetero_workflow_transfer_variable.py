#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from federatedml.util.transfer_variable.base_transfer_variable import BaseTransferVariable, Variable


class HeteroWorkFlowTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.train_data = Variable(name="HeteroWorkFlowTransferVariable.train_data", auth={'src': "guest", 'dst': ['host']})
        self.test_data = Variable(name="HeteroWorkFlowTransferVariable.test_data", auth={'src': "guest", 'dst': ['host']})
        pass
