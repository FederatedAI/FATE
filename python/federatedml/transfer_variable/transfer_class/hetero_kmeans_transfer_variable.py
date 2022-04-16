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
class HeteroKmeansTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.arbiter_tol = self._create_variable(name='arbiter_tol', src=['arbiter'], dst=['host', 'guest'])
        self.cluster_result = self._create_variable(name='cluster_result', src=['arbiter'], dst=['host', 'guest'])
        self.cluster_evaluation = self._create_variable(
            name='cluster_evaluation', src=['arbiter'], dst=['host', 'guest'])
        self.guest_dist = self._create_variable(name='guest_dist', src=['guest'], dst=['arbiter'])
        self.guest_tol = self._create_variable(name='guest_tol', src=['guest'], dst=['arbiter'])
        self.host_dist = self._create_variable(name='host_dist', src=['host'], dst=['arbiter'])
        self.host_tol = self._create_variable(name='host_tol', src=['host'], dst=['arbiter'])
        self.centroid_list = self._create_variable(name='centroid_list', src=['guest'], dst=['host'])
