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

from federatedml.util.transfer_variable.base_transfer_variable import BaseTransferVariable, Variable
from federatedml.util.transfer_variable.base_transfer_variable import Variable


class HeteroFTLTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.paillier_pubkey = Variable(name="HeteroFTLTransferVariable.paillier_pubkey", auth={'src': "arbiter", 'dst': ['host', 'guest']})
        self.batch_data_index = Variable(name="HeteroFTLTransferVariable.batch_data_index", auth={'src': "guest", 'dst': ['host']})
        self.host_component_list = Variable(name="HeteroFTLTransferVariable.host_component_list", auth={'src': "host", 'dst': ['guest']})
        self.guest_component_list = Variable(name="HeteroFTLTransferVariable.guest_component_list", auth={'src': "guest", 'dst': ['host']})
        self.host_precomputed_comp_list = Variable(name="HeteroFTLTransferVariable.host_precomputed_comp_list", auth={'src': "host", 'dst': ['guest']})
        self.guest_precomputed_comp_list = Variable(name="HeteroFTLTransferVariable.guest_precomputed_comp_list", auth={'src': "guest", 'dst': ['host']})
        self.encrypt_guest_gradient = Variable(name="HeteroFTLTransferVariable.encrypt_guest_gradient", auth={'src': "guest", 'dst': ['arbiter']})
        self.decrypt_guest_gradient = Variable(name="HeteroFTLTransferVariable.decrypt_guest_gradient", auth={'src': "arbiter", 'dst': ['guest']})
        self.encrypt_host_gradient = Variable(name="HeteroFTLTransferVariable.encrypt_host_gradient", auth={'src': "host", 'dst': ['arbiter']})
        self.decrypt_host_gradient = Variable(name="HeteroFTLTransferVariable.decrypt_host_gradient", auth={'src': "arbiter", 'dst': ['host']})
        self.encrypt_loss = Variable(name="HeteroFTLTransferVariable.encrypt_loss", auth={'src': "guest", 'dst': ['arbiter']})
        self.is_encrypted_ftl_stopped = Variable(name="HeteroFTLTransferVariable.is_encrypted_ftl_stopped", auth={'src': "arbiter", 'dst': ['host', 'guest']})
        self.is_stopped = Variable(name="HeteroFTLTransferVariable.is_stopped", auth={'src': "guest", 'dst': ['host']})
        self.batch_size = Variable(name="HeteroFTLTransferVariable.batch_size", auth={'src': "guest", 'dst': ['host']})
        self.batch_num = Variable(name="HeteroFTLTransferVariable.batch_num", auth={'src': "guest", 'dst': ['arbiter', 'host']})
        self.host_prob = Variable(name="HeteroFTLTransferVariable.host_prob", auth={'src': "host", 'dst': ['guest']})
        self.pred_prob = Variable(name="HeteroFTLTransferVariable.pred_prob", auth={'src': "guest", 'dst': ['host']})
        self.encrypt_prob = Variable(name="HeteroFTLTransferVariable.encrypt_prob", auth={'src': "guest", 'dst': ['arbiter']})
        self.decrypt_prob = Variable(name="HeteroFTLTransferVariable.decrypt_prob", auth={'src': "arbiter", 'dst': ['guest']})
        self.guest_sample_indexes = Variable(name="HeteroFTLTransferVariable.guest_sample_indexes", auth={'src': "guest", 'dst': ['host']})
        self.host_sample_indexes = Variable(name="HeteroFTLTransferVariable.host_sample_indexes", auth={'src': "host", 'dst': ['guest']})
        self.guest_public_key = Variable(name="HeteroFTLTransferVariable.guest_public_key", auth={'src': "guest", 'dst': ['host']})
        self.host_public_key = Variable(name="HeteroFTLTransferVariable.host_public_key", auth={'src': "host", 'dst': ['guest']})
        self.masked_enc_guest_gradients = Variable(name="HeteroFTLTransferVariable.masked_enc_guest_gradients", auth={'src': "guest", 'dst': ['host']})
        self.masked_enc_host_gradients = Variable(name="HeteroFTLTransferVariable.masked_enc_host_gradients", auth={'src': "host", 'dst': ['guest']})
        self.masked_dec_guest_gradients = Variable(name="HeteroFTLTransferVariable.masked_dec_guest_gradients", auth={'src': "host", 'dst': ['guest']})
        self.masked_dec_host_gradients = Variable(name="HeteroFTLTransferVariable.masked_dec_host_gradients", auth={'src': "guest", 'dst': ['host']})
        self.masked_enc_loss = Variable(name="HeteroFTLTransferVariable.masked_enc_loss", auth={'src': "guest", 'dst': ['host']})
        self.masked_dec_loss = Variable(name="HeteroFTLTransferVariable.masked_dec_loss", auth={'src': "host", 'dst': ['guest']})
        self.is_decentralized_enc_ftl_stopped = Variable(name="HeteroFTLTransferVariable.is_decentralized_enc_ftl_stopped", auth={'src': "guest", 'dst': ['host']})
        pass
