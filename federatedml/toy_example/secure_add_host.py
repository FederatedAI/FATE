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

from arch.api import federation
from federatedml.util.transfer_variable import SecureAddExampleTransferVariable
import numpy as np


class SecureAddHost(object):
    def __init__(self, y):
        self.y = y
        self.y1 = None
        self.y2 = None
        self.x2 = None
        self.x2_plus_y2 = None
        self.transfer_inst = SecureAddExampleTransferVariable()

    def share(self, y):
        first = np.random.uniform(y, -y)
        return first, y - first

    def secure(self):
        y_shares = self.y.mapValues(self.share)
        self.y1 = y_shares.mapValues(lambda shares: shares[0])
        self.y2 = y_shares.mapValues(lambda shares: shares[1])

    def add(self):
        self.x2_plus_y2 = self.y2.join(self.x2, lambda y, x: y + x)
        host_sum = self.x2_plus_y2.reduce(lambda x, y: x + y)
        return host_sum

    def sync_share_to_guest(self):
        federation.remote(obj=self.y1,
                          name=self.transfer_inst.host_share.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.host_share),
                          role="guest",
                          idx=0)

    def recv_share_from_guest(self):
        self.x2 = federation.get(name=self.transfer_inst.guest_share.name,
                                 tag=self.transfer_inst.generate_transferid(self.transfer_inst.guest_share),
                                 idx=0)

    def sync_host_sum_to_guest(self, host_sum):
        federation.remote(obj=host_sum,
                          name=self.transfer_inst.host_sum.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.host_sum),
                          role="guest",
                          idx=0)

    def run(self):
        self.secure()
        self.recv_share_from_guest()
        self.sync_share_to_guest()
        host_sum = self.add()
        self.sync_host_sum_to_guest(host_sum)


