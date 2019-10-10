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
class HeteroFTLTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.paillier_pubkey = Variable(name='HeteroFTLTransferVariable.paillier_pubkey', auth=dict(src='arbiter', dst=['host', 'guest']), transfer_variable=self)
        self.batch_data_index = Variable(name='HeteroFTLTransferVariable.batch_data_index', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.host_component_list = Variable(name='HeteroFTLTransferVariable.host_component_list', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.guest_component_list = Variable(name='HeteroFTLTransferVariable.guest_component_list', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.host_precomputed_comp_list = Variable(name='HeteroFTLTransferVariable.host_precomputed_comp_list', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.guest_precomputed_comp_list = Variable(name='HeteroFTLTransferVariable.guest_precomputed_comp_list', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.encrypt_guest_gradient = Variable(name='HeteroFTLTransferVariable.encrypt_guest_gradient', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.decrypt_guest_gradient = Variable(name='HeteroFTLTransferVariable.decrypt_guest_gradient', auth=dict(src='arbiter', dst=['guest']), transfer_variable=self)
        self.encrypt_host_gradient = Variable(name='HeteroFTLTransferVariable.encrypt_host_gradient', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.decrypt_host_gradient = Variable(name='HeteroFTLTransferVariable.decrypt_host_gradient', auth=dict(src='arbiter', dst=['host']), transfer_variable=self)
        self.encrypt_loss = Variable(name='HeteroFTLTransferVariable.encrypt_loss', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.is_encrypted_ftl_stopped = Variable(name='HeteroFTLTransferVariable.is_encrypted_ftl_stopped', auth=dict(src='arbiter', dst=['host', 'guest']), transfer_variable=self)
        self.is_stopped = Variable(name='HeteroFTLTransferVariable.is_stopped', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.batch_size = Variable(name='HeteroFTLTransferVariable.batch_size', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.batch_num = Variable(name='HeteroFTLTransferVariable.batch_num', auth=dict(src='guest', dst=['arbiter', 'host']), transfer_variable=self)
        self.host_prob = Variable(name='HeteroFTLTransferVariable.host_prob', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.pred_prob = Variable(name='HeteroFTLTransferVariable.pred_prob', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.encrypt_prob = Variable(name='HeteroFTLTransferVariable.encrypt_prob', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.decrypt_prob = Variable(name='HeteroFTLTransferVariable.decrypt_prob', auth=dict(src='arbiter', dst=['guest']), transfer_variable=self)
        self.guest_sample_indexes = Variable(name='HeteroFTLTransferVariable.guest_sample_indexes', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.host_sample_indexes = Variable(name='HeteroFTLTransferVariable.host_sample_indexes', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.guest_public_key = Variable(name='HeteroFTLTransferVariable.guest_public_key', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.host_public_key = Variable(name='HeteroFTLTransferVariable.host_public_key', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.masked_enc_guest_gradients = Variable(name='HeteroFTLTransferVariable.masked_enc_guest_gradients', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.masked_enc_host_gradients = Variable(name='HeteroFTLTransferVariable.masked_enc_host_gradients', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.masked_dec_guest_gradients = Variable(name='HeteroFTLTransferVariable.masked_dec_guest_gradients', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.masked_dec_host_gradients = Variable(name='HeteroFTLTransferVariable.masked_dec_host_gradients', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.masked_enc_loss = Variable(name='HeteroFTLTransferVariable.masked_enc_loss', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.masked_dec_loss = Variable(name='HeteroFTLTransferVariable.masked_dec_loss', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.is_decentralized_enc_ftl_stopped = Variable(name='HeteroFTLTransferVariable.is_decentralized_enc_ftl_stopped', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        pass
