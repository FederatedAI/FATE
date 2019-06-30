#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from federatedml.util.transfer_variable.base_transfer_variable import BaseTransferVariable, Variable


class SampleTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.sample_ids = Variable(name="SampleTransferVariable.sample_ids", auth={'src': "guest", 'dst': ['host']})
        pass
