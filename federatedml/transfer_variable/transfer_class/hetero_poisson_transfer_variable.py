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
class HeteroPoissonTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.batch_data_index = self._create_variable(name='batch_data_index')
        self.batch_info = self._create_variable(name='batch_info')
        self.converge_flag = self._create_variable(name='converge_flag')
        self.fore_gradient = self._create_variable(name='fore_gradient')
        self.guest_gradient = self._create_variable(name='guest_gradient')
        self.guest_optim_gradient = self._create_variable(name='guest_optim_gradient')
        self.host_forward = self._create_variable(name='host_forward')
        self.host_gradient = self._create_variable(name='host_gradient')
        self.host_loss_regular = self._create_variable(name='host_loss_regular')
        self.host_optim_gradient = self._create_variable(name='host_optim_gradient')
        self.host_partial_prediction = self._create_variable(name='host_partial_prediction')
        self.loss = self._create_variable(name='loss')
        self.loss_intermediate = self._create_variable(name='loss_intermediate')
        self.paillier_pubkey = self._create_variable(name='paillier_pubkey')
