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
class HeteroLRTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.paillier_pubkey = Variable(name='HeteroLRTransferVariable.paillier_pubkey', auth=dict(src='arbiter', dst=['host', 'guest']), transfer_variable=self)
        self.batch_data_index = Variable(name='HeteroLRTransferVariable.batch_data_index', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.host_forward_dict = Variable(name='HeteroLRTransferVariable.host_forward_dict', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.fore_gradient = Variable(name='HeteroLRTransferVariable.fore_gradient', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.guest_gradient = Variable(name='HeteroLRTransferVariable.guest_gradient', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.guest_optim_gradient = Variable(name='HeteroLRTransferVariable.guest_optim_gradient', auth=dict(src='arbiter', dst=['guest']), transfer_variable=self)
        self.host_loss_regular = Variable(name='HeteroLRTransferVariable.host_loss_regular', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.loss = Variable(name='HeteroLRTransferVariable.loss', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.loss_intermediate = Variable(name='HeteroLRTransferVariable.loss_intermediate', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.converge_flag = Variable(name='HeteroLRTransferVariable.converge_flag', auth=dict(src='arbiter', dst=['host', 'guest']), transfer_variable=self)
        self.batch_info = Variable(name='HeteroLRTransferVariable.batch_info', auth=dict(src='guest', dst=['host', 'arbiter']), transfer_variable=self)
        self.host_optim_gradient = Variable(name='HeteroLRTransferVariable.host_optim_gradient', auth=dict(src='arbiter', dst=['host']), transfer_variable=self)
        self.host_gradient = Variable(name='HeteroLRTransferVariable.host_gradient', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.host_prob = Variable(name='HeteroLRTransferVariable.host_prob', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        pass
