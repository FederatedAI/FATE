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


class SecureAddGuest(object):
    def __init__(self, x):
        self.x = x
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.x1_plus_y1 = None
        self.transfer_inst = SecureAddExampleTransferVariable()

    def share(self, x):
        first = np.random.uniform(x, -x)
        return first, x - first

    def secure(self):
        x_shares = self.x.mapValues(self.share)
        self.x1 = x_shares.mapValues(lambda shares: shares[0])
        self.x2 = x_shares.mapValues(lambda shares: shares[1])

    def add(self):
        self.x1_plus_y1 = self.x1.join(self.y1, lambda x, y: x + y)
        guest_sum = self.x1_plus_y1.reduce(lambda x, y: x + y)
        return guest_sum

    def reconstruct(self, guest_sum, host_sum):
        print ("host sum is %.4f" % host_sum)
        print ("guest sum is %.4f" % guest_sum)
        secure_sum = host_sum + guest_sum

        print ("Secure Add Result is %.4f" % secure_sum)

        return secure_sum

    def sync_share_to_host(self):
        federation.remote(obj=self.x2,
                          name=self.transfer_inst.guest_share.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.guest_share),
                          role="host",
                          idx=0)

    def recv_share_from_host(self):
        self.y1 = federation.get(name=self.transfer_inst.host_share.name,
                                 tag=self.transfer_inst.generate_transferid(self.transfer_inst.host_share),
                                 idx=0)

    def recv_host_sum_from_host(self):
        host_sum = federation.get(name=self.transfer_inst.host_sum.name,
                                  tag=self.transfer_inst.generate_transferid(self.transfer_inst.host_sum),
                                  idx=0)

        return host_sum

    def run(self):
        self.secure()
        self.sync_share_to_host()
        self.recv_share_from_host()
        
        guest_sum = self.add()
        host_sum = self.recv_host_sum_from_host()
        secure_sum = self.reconstruct(guest_sum, host_sum)

        return secure_sum


