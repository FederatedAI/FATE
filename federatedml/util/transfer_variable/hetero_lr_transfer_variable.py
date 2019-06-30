#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from federatedml.util.transfer_variable.base_transfer_variable import BaseTransferVariable, Variable


class HeteroLRTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.paillier_pubkey = Variable(name="HeteroLRTransferVariable.paillier_pubkey", auth={'src': "arbiter", 'dst': ['host', 'guest']})
        self.batch_data_index = Variable(name="HeteroLRTransferVariable.batch_data_index", auth={'src': "guest", 'dst': ['host']})
        self.host_forward_dict = Variable(name="HeteroLRTransferVariable.host_forward_dict", auth={'src': "host", 'dst': ['guest']})
        self.fore_gradient = Variable(name="HeteroLRTransferVariable.fore_gradient", auth={'src': "guest", 'dst': ['host']})
        self.guest_gradient = Variable(name="HeteroLRTransferVariable.guest_gradient", auth={'src': "guest", 'dst': ['arbiter']})
        self.guest_optim_gradient = Variable(name="HeteroLRTransferVariable.guest_optim_gradient", auth={'src': "arbiter", 'dst': ['guest']})
        self.host_loss_regular = Variable(name="HeteroLRTransferVariable.host_loss_regular", auth={'src': "host", 'dst': ['guest']})
        self.loss = Variable(name="HeteroLRTransferVariable.loss", auth={'src': "guest", 'dst': ['arbiter']})
        self.is_stopped = Variable(name="HeteroLRTransferVariable.is_stopped", auth={'src': "arbiter", 'dst': ['host', 'guest']})
        self.batch_info = Variable(name="HeteroLRTransferVariable.batch_info", auth={'src': "guest", 'dst': ['host', 'arbiter']})
        self.host_optim_gradient = Variable(name="HeteroLRTransferVariable.host_optim_gradient", auth={'src': "arbiter", 'dst': ['host']})
        self.host_gradient = Variable(name="HeteroLRTransferVariable.host_gradient", auth={'src': "host", 'dst': ['arbiter']})
        self.host_prob = Variable(name="HeteroLRTransferVariable.host_prob", auth={'src': "host", 'dst': ['guest']})
        pass
