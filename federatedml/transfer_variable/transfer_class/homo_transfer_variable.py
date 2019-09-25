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
class HomoTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.guest_uuid = Variable(name='HomoTransferVariable.guest_uuid', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.host_uuid = Variable(name='HomoTransferVariable.host_uuid', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.uuid_conflict_flag = Variable(name='HomoTransferVariable.uuid_conflict_flag', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.dh_pubkey = Variable(name='HomoTransferVariable.dh_pubkey', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.dh_ciphertext_host = Variable(name='HomoTransferVariable.dh_ciphertext_host', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.dh_ciphertext_guest = Variable(name='HomoTransferVariable.dh_ciphertext_guest', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.dh_ciphertext_bc = Variable(name='HomoTransferVariable.dh_ciphertext_bc', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.guest_party_weight = Variable(name='HomoTransferVariable.guest_party_weight', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.host_party_weight = Variable(name='HomoTransferVariable.host_party_weight', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.use_encrypt = Variable(name='HomoTransferVariable.use_encrypt', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.paillier_pubkey = Variable(name='HomoTransferVariable.paillier_pubkey', auth=dict(src='arbiter', dst=['host']), transfer_variable=self)
        self.re_encrypt_times = Variable(name='HomoTransferVariable.re_encrypt_times', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.model_to_re_encrypt = Variable(name='HomoTransferVariable.model_to_re_encrypt', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.model_re_encrypted = Variable(name='HomoTransferVariable.model_re_encrypted', auth=dict(src='arbiter', dst=['host']), transfer_variable=self)
        self.is_converge = Variable(name='HomoTransferVariable.is_converge', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.host_loss = Variable(name='HomoTransferVariable.host_loss', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.guest_loss = Variable(name='HomoTransferVariable.guest_loss', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.aggregated_model = Variable(name='HomoTransferVariable.aggregated_model', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.host_model = Variable(name='HomoTransferVariable.host_model', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.guest_model = Variable(name='HomoTransferVariable.guest_model', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.to_encrypt_model = Variable(name='HomoTransferVariable.to_encrypt_model', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.re_encrypted_model = Variable(name='HomoTransferVariable.re_encrypted_model', auth=dict(src='arbiter', dst=['host']), transfer_variable=self)
        pass
