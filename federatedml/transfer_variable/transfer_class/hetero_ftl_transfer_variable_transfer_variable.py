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

from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables


# noinspection PyAttributeOutsideInit
class HeteroFTLTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.batch_data_index = self._create_variable(name='batch_data_index', src=['guest'], dst=['host'])
        self.batch_num = self._create_variable(name='batch_num', src=['guest'], dst=['arbiter', 'host'])
        self.batch_size = self._create_variable(name='batch_size', src=['guest'], dst=['host'])
        self.decrypt_guest_gradient = self._create_variable(name='decrypt_guest_gradient', src=['arbiter'], dst=['guest'])
        self.decrypt_host_gradient = self._create_variable(name='decrypt_host_gradient', src=['arbiter'], dst=['host'])
        self.decrypt_prob = self._create_variable(name='decrypt_prob', src=['arbiter'], dst=['guest'])
        self.encrypt_guest_gradient = self._create_variable(name='encrypt_guest_gradient', src=['guest'], dst=['arbiter'])
        self.encrypt_host_gradient = self._create_variable(name='encrypt_host_gradient', src=['host'], dst=['arbiter'])
        self.encrypt_loss = self._create_variable(name='encrypt_loss', src=['guest'], dst=['arbiter'])
        self.encrypt_prob = self._create_variable(name='encrypt_prob', src=['guest'], dst=['arbiter'])
        self.guest_component_list = self._create_variable(name='guest_component_list', src=['guest'], dst=['host'])
        self.guest_precomputed_comp_list = self._create_variable(name='guest_precomputed_comp_list', src=['guest'], dst=['host'])
        self.guest_public_key = self._create_variable(name='guest_public_key', src=['guest'], dst=['host'])
        self.guest_sample_indexes = self._create_variable(name='guest_sample_indexes', src=['guest'], dst=['host'])
        self.host_component_list = self._create_variable(name='host_component_list', src=['host'], dst=['guest'])
        self.host_precomputed_comp_list = self._create_variable(name='host_precomputed_comp_list', src=['host'], dst=['guest'])
        self.host_prob = self._create_variable(name='host_prob', src=['host'], dst=['guest'])
        self.host_public_key = self._create_variable(name='host_public_key', src=['host'], dst=['guest'])
        self.host_sample_indexes = self._create_variable(name='host_sample_indexes', src=['host'], dst=['guest'])
        self.is_decentralized_enc_ftl_stopped = self._create_variable(name='is_decentralized_enc_ftl_stopped', src=['guest'], dst=['host'])
        self.is_encrypted_ftl_stopped = self._create_variable(name='is_encrypted_ftl_stopped', src=['arbiter'], dst=['host', 'guest'])
        self.is_stopped = self._create_variable(name='is_stopped', src=['guest'], dst=['host'])
        self.masked_dec_guest_gradients = self._create_variable(name='masked_dec_guest_gradients', src=['host'], dst=['guest'])
        self.masked_dec_host_gradients = self._create_variable(name='masked_dec_host_gradients', src=['guest'], dst=['host'])
        self.masked_dec_loss = self._create_variable(name='masked_dec_loss', src=['host'], dst=['guest'])
        self.masked_enc_guest_gradients = self._create_variable(name='masked_enc_guest_gradients', src=['guest'], dst=['host'])
        self.masked_enc_host_gradients = self._create_variable(name='masked_enc_host_gradients', src=['host'], dst=['guest'])
        self.masked_enc_loss = self._create_variable(name='masked_enc_loss', src=['guest'], dst=['host'])
        self.paillier_pubkey = self._create_variable(name='paillier_pubkey', src=['arbiter'], dst=['host', 'guest'])
        self.pred_prob = self._create_variable(name='pred_prob', src=['guest'], dst=['host'])
