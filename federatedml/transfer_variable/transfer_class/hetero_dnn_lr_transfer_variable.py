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
class HeteroDNNLRTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.guest_dec_gradient = Variable(name='HeteroDNNLRTransferVariable.guest_dec_gradient', auth=dict(src='arbiter', dst=['guest']), transfer_variable=self)
        self.guest_enc_gradient = Variable(name='HeteroDNNLRTransferVariable.guest_enc_gradient', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.host_dec_gradient = Variable(name='HeteroDNNLRTransferVariable.host_dec_gradient', auth=dict(src='arbiter', dst=['host']), transfer_variable=self)
        self.host_enc_gradient = Variable(name='HeteroDNNLRTransferVariable.host_enc_gradient', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        pass
