#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from federatedml.util.transfer_variable.base_transfer_variable import BaseTransferVariable, Variable


class SecureAddExampleTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.guest_share = Variable(name="SecureAddExampleTransferVariable.guest_share", auth={'src': "guest", 'dst': ['host']})
        self.host_share = Variable(name="SecureAddExampleTransferVariable.host_share", auth={'src': "host", 'dst': ['guest']})
        self.host_sum = Variable(name="SecureAddExampleTransferVariable.host_sum", auth={'src': "host", 'dst': ['guest']})
        pass
