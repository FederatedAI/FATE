#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from federatedml.util.transfer_variable.base_transfer_variable import BaseTransferVariable, Variable


class HeteroDNNLRTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.guest_dec_gradient = Variable(name="HeteroDNNLRTransferVariable.guest_dec_gradient", auth={'src': "arbiter", 'dst': ['guest']})
        self.guest_enc_gradient = Variable(name="HeteroDNNLRTransferVariable.guest_enc_gradient", auth={'src': "guest", 'dst': ['arbiter']})
        self.host_dec_gradient = Variable(name="HeteroDNNLRTransferVariable.host_dec_gradient", auth={'src': "arbiter", 'dst': ['host']})
        self.host_enc_gradient = Variable(name="HeteroDNNLRTransferVariable.host_enc_gradient", auth={'src': "host", 'dst': ['arbiter']})
        pass
