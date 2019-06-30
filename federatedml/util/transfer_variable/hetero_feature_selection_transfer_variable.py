#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from federatedml.util.transfer_variable.base_transfer_variable import BaseTransferVariable, Variable


class HeteroFeatureSelectionTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.result_left_cols = Variable(name="HeteroFeatureSelectionTransferVariable.result_left_cols", auth={'src': "guest", 'dst': ['host']})
        self.host_iv_threshold = Variable(name="HeteroFeatureSelectionTransferVariable.host_iv_threshold", auth={'src': "host", 'dst': ['guest']})
        pass
