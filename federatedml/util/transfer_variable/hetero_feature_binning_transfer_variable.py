#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from federatedml.util.transfer_variable.base_transfer_variable import BaseTransferVariable, Variable


class HeteroFeatureBinningTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.paillier_pubkey = Variable(name="HeteroFeatureBinningTransferVariable.paillier_pubkey", auth={'src': "guest", 'dst': ['host']})
        self.encrypted_label = Variable(name="HeteroFeatureBinningTransferVariable.encrypted_label", auth={'src': "guest", 'dst': ['host']})
        self.encrypted_bin_sum = Variable(name="HeteroFeatureBinningTransferVariable.encrypted_bin_sum", auth={'src': "host", 'dst': ['guest']})
        pass
