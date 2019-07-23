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


class HeteroFeatureBinningTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.paillier_pubkey = Variable(name="HeteroFeatureBinningTransferVariable.paillier_pubkey", auth={'src': "guest", 'dst': ['host']})
        self.encrypted_label = Variable(name="HeteroFeatureBinningTransferVariable.encrypted_label", auth={'src': "guest", 'dst': ['host']})
        self.encrypted_bin_sum = Variable(name="HeteroFeatureBinningTransferVariable.encrypted_bin_sum", auth={'src': "host", 'dst': ['guest']})
        pass
