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
class FTLTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.guest_components = self._create_variable(name='guest_components', src=['guest'], dst=['host'])
        self.y_overlap_2_phi_2 = self._create_variable(name='y_overlap_2_phi_2', src=['guest'], dst=['host'])
        self.y_overlap_phi = self._create_variable(name='y_overlap_phi', src=['guest'], dst=['host'])
        self.mapping_comp_a = self._create_variable(name='mapping_comp_a', src=['guest'], dst=['host'])
        self.stop_flag = self._create_variable(name='stop_flag', src=['guest'], dst=['host'])
        self.host_components = self._create_variable(name='host_components', src=['host'], dst=['guest'])
        self.overlap_ub = self._create_variable(name='overlap_ub', src=['host'], dst=['guest'])
        self.overlap_ub_2 = self._create_variable(name='overlap_ub_2', src=['host'], dst=['guest'])
        self.mapping_comp_b = self._create_variable(name='mapping_comp_b', src=['host'], dst=['guest'])
        self.host_side_gradients = self._create_variable(name='host_side_gradients', src=['host'], dst=['guest'])
        self.guest_side_gradients = self._create_variable(name='guest_side_gradients', src=['guest'], dst=['host'])
        self.guest_side_const = self._create_variable(name='guest_side_const', src=['guest'], dst=['host'])
        self.encrypted_loss = self._create_variable(name='encrypted_loss', src=['guest'], dst=['host'])
        self.decrypted_loss = self._create_variable(name='decrypted_loss', src=['host'], dst=['guest'])
        self.decrypted_guest_gradients = self._create_variable(
            name='decrypted_guest_gradients', src=['host'], dst=['guest'])
        self.decrypted_guest_const = self._create_variable(name='decrypted_guest_const', src=['host'], dst=['guest'])
        self.decrypted_host_gradients = self._create_variable(
            name='decrypted_host_gradients', src=['guest'], dst=['host'])
        self.predict_stop_flag = self._create_variable(name='predict_stop_flag', src=['host'], dst=['guest'])
        self.predict_host_u = self._create_variable(name='predict_host_u', src=['host'], dst=['guest'])
        self.encrypted_predict_score = self._create_variable(
            name='encrypted_predict_score', src=['guest'], dst=['host'])
        self.masked_predict_score = self._create_variable(name='masked_predict_score', src=['host'], dst=['guest'])
        self.final_predict_score = self._create_variable(name='final_predict_score', src=['guest'], dst=['host'])
        self.predict_batch_num = self._create_variable(name='predict_batch_num', src=['host'], dst=['guest'])
