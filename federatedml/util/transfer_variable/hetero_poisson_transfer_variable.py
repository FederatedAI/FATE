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


class HeteroPoissonTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.paillier_pubkey = Variable(name="HeteroPoissonTransferVariable.paillier_pubkey", auth={'src': "arbiter", 'dst': ['host', 'guest']})
        self.batch_data_index = Variable(name="HeteroPoissonTransferVariable.batch_data_index", auth={'src': "guest", 'dst': ['host']})
        self.host_forward_wx = Variable(name="HeteroPoissonTransferVariable.host_forward", auth={'src': "host", 'dst': ['guest']})
        self.host_loss = Variable(name="HeteroPoissonTransferVariable.host_loss", auth={'src': "host", 'dst': ['guest']})
        self.fore_gradient = Variable(name="HeteroPoissonTransferVariable.fore_gradient", auth={'src': "guest", 'dst': ['host']})
        self.guest_gradient = Variable(name="HeteroPoissonTransferVariable.guest_gradient", auth={'src': "guest", 'dst': ['arbiter']})
        self.optim_guest_gradient = Variable(name="HeteroPoissonTransferVariable.optim_guest_gradient", auth={'src': "arbiter", 'dst': ['guest']})
        self.loss = Variable(name="HeteroPoissonTransferVariable.loss", auth={'src': "guest", 'dst': ['arbiter']})
        self.is_stopped = Variable(name="HeteroPoissonTransferVariable.is_stopped", auth={'src': "arbiter", 'dst': ['host', 'guest']})
        self.batch_info = Variable(name="HeteroPoissonTransferVariable.batch_info", auth={'src': "guest", 'dst': ['host', 'arbiter']})
        self.optim_host_gradient = Variable(name="HeteroPoissonTransferVariable.optim_host_gradient", auth={'src': "arbiter", 'dst': ['host']})
        self.host_gradient = Variable(name="HeteroPoissonTransferVariable.host_gradient", auth={'src': "host", 'dst': ['arbiter']})
        self.host_partial_prediction = Variable(name="HeteroPoissonTransferVariable.host_partial_prediction", auth={'src': "host", 'dst': ['guest']})
        pass
