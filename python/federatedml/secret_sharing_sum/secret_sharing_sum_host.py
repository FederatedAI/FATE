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


from federatedml.util import LOGGER
from federatedml.param.secure_sharing_sum_param import SecureSharingSumParam
from federatedml.transfer_variable.transfer_class.secret_sharing_sum_transfer_variable import \
    SecretSharingSumTransferVariables
from federatedml.secret_sharing_sum.base_secret_sharing_sum import BaseSecretSharingSum


class SecretSharingSumHost(BaseSecretSharingSum):
    def __init__(self):
        super(SecretSharingSumHost, self).__init__()
        self.transfer_inst = SecretSharingSumTransferVariables()
        self.model_param = SecureSharingSumParam()
        self.host_party_idlist = []
        self.local_partyid = -1

    def _init_model(self, model_param: SecureSharingSumParam):
        self.need_verify = model_param.need_verify

    def _init_data(self, data_inst):
        self.local_partyid = self.component_properties.local_partyid
        self.host_party_idlist = self.component_properties.host_party_idlist
        self.host_count = len(self.host_party_idlist)
        self.vss.set_share_amount(self.host_count+1)
        self.x = data_inst

    def recv_primes_from_guest(self):
        prime = self.transfer_inst.guest_share_primes.get(idx=0)
        self.vss.set_prime(prime)

    def sync_share_to_parties(self):
        for idx, party_id in enumerate(self.host_party_idlist):
            if self.local_partyid != party_id:
                self.transfer_inst.host_share_to_host.remote(self.secret_sharing[idx],
                                                             role="host",
                                                             idx=idx)
            else:
                self.x_plus_y = self.secret_sharing[idx]
                self.transfer_inst.host_share_to_guest.remote(self.secret_sharing[-1],
                                                              role="guest",
                                                              idx=0)

    def recv_share_from_parties(self):
        for idx, party_id in enumerate(self.host_party_idlist):
            if self.local_partyid != party_id:
                self.y_recv.append(self.transfer_inst.host_share_to_host.get(idx=idx))
            else:
                self.y_recv.append(self.transfer_inst.guest_share_secret.get(idx=0))

    def sync_host_sum_to_guest(self):
        self.transfer_inst.host_sum.remote(self.x_plus_y,
                                           role="guest",
                                           idx=-1)

    def fit(self, data_inst):

        LOGGER.info("begin to make host data")
        self._init_data(data_inst)

        LOGGER.info("get primes from host")
        self.recv_primes_from_guest()

        LOGGER.info("split data into multiple random parts")
        self.secure()

        LOGGER.info("share one of random part data to multiple parties")
        self.sync_share_to_parties()

        LOGGER.info("get share of one random part data from multiple parties")
        self.recv_share_from_parties()

        LOGGER.info("begin to get sum of host and guest")
        self.sharing_sum()

        LOGGER.info("send host sum to guest")
        self.sync_host_sum_to_guest()
