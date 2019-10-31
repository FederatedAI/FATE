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
class HomoLRTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.guest_uuid = Variable(name='HomoLRTransferVariable.guest_uuid', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.host_uuid = Variable(name='HomoLRTransferVariable.host_uuid', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.uuid_conflict_flag = Variable(name='HomoLRTransferVariable.uuid_conflict_flag', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.dh_pubkey = Variable(name='HomoLRTransferVariable.dh_pubkey', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.dh_ciphertext_host = Variable(name='HomoLRTransferVariable.dh_ciphertext_host', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.dh_ciphertext_guest = Variable(name='HomoLRTransferVariable.dh_ciphertext_guest', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.dh_ciphertext_bc = Variable(name='HomoLRTransferVariable.dh_ciphertext_bc', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.paillier_pubkey = Variable(name='HomoLRTransferVariable.paillier_pubkey', auth=dict(src='arbiter', dst=['host']), transfer_variable=self)
        self.guest_model = Variable(name='HomoLRTransferVariable.guest_model', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.host_model = Variable(name='HomoLRTransferVariable.host_model', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.aggregated_model = Variable(name='HomoLRTransferVariable.aggregated_model', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.to_encrypt_model = Variable(name='HomoLRTransferVariable.to_encrypt_model', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.re_encrypted_model = Variable(name='HomoLRTransferVariable.re_encrypted_model', auth=dict(src='arbiter', dst=['host']), transfer_variable=self)
        self.re_encrypt_times = Variable(name='HomoLRTransferVariable.re_encrypt_times', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.is_converge = Variable(name='HomoLRTransferVariable.is_converge', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.guest_loss = Variable(name='HomoLRTransferVariable.guest_loss', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.host_loss = Variable(name='HomoLRTransferVariable.host_loss', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.use_encrypt = Variable(name='HomoLRTransferVariable.use_encrypt', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.guest_party_weight = Variable(name='HomoLRTransferVariable.guest_party_weight', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.host_party_weight = Variable(name='HomoLRTransferVariable.host_party_weight', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.predict_wx = Variable(name='HomoLRTransferVariable.predict_wx', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.predict_result = Variable(name='HomoLRTransferVariable.predict_result', auth=dict(src='arbiter', dst=['host']), transfer_variable=self)
        pass
