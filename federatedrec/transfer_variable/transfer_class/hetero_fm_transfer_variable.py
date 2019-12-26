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
class HeteroFMTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.paillier_pubkey = Variable(name='HeteroFMTransferVariable.paillier_pubkey', auth=dict(src='arbiter', dst=['host', 'guest']), transfer_variable=self)
        self.batch_data_index = Variable(name='HeteroFMTransferVariable.batch_data_index', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.host_forward_dict = Variable(name='HeteroFMTransferVariable.host_forward_dict', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.fore_gradient = Variable(name='HeteroFMTransferVariable.fore_gradient', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.guest_gradient = Variable(name='HeteroFMTransferVariable.guest_gradient', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.guest_optim_gradient = Variable(name='HeteroFMTransferVariable.guest_optim_gradient', auth=dict(src='arbiter', dst=['guest']), transfer_variable=self)
        self.host_loss_regular = Variable(name='HeteroFMTransferVariable.host_loss_regular', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.loss = Variable(name='HeteroFMTransferVariable.loss', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.loss_intermediate = Variable(name='HeteroFMTransferVariable.loss_intermediate', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.converge_flag = Variable(name='HeteroFMTransferVariable.converge_flag', auth=dict(src='arbiter', dst=['host', 'guest']), transfer_variable=self)
        self.batch_info = Variable(name='HeteroFMTransferVariable.batch_info', auth=dict(src='guest', dst=['host', 'arbiter']), transfer_variable=self)
        self.host_optim_gradient = Variable(name='HeteroFMTransferVariable.host_optim_gradient', auth=dict(src='arbiter', dst=['host']), transfer_variable=self)
        self.host_gradient = Variable(name='HeteroFMTransferVariable.host_gradient', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.host_prob = Variable(name='HeteroFMTransferVariable.host_prob', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.agg_vx_mul_fg = Variable(name='HeteroFMTransferVariable.agg_vx_mul_fg', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.aggregated_forwards = Variable(name='HeteroFMTransferVariable.aggregated_forwards', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        pass
