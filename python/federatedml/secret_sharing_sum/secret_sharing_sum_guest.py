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
from federatedml.transfer_variable.transfer_class import secret_sharing_sum_transfer_variable
from federatedml.secret_sharing_sum.base_secret_sharing_sum import BaseSecretSharingSum


class SecretSharingSumGuest(BaseSecretSharingSum):
    def __init__(self):
        super(SecretSharingSumGuest, self).__init__()
        self.transfer_inst = secret_sharing_sum_transfer_variable.SecretSharingSumTransferVariables()
        self.model_param = SecureSharingSumParam()

    def _init_model(self, model_param: SecureSharingSumParam):
        self.need_verify = model_param.need_verify

    def _init_data(self, data_inst):
        self.share_amount = len(self.component_properties.host_party_idlist)+1
        self.vss.set_share_amount(self.share_amount)
        self.vss.generate_prime()
        self.x = data_inst

    def sync_primes_to_host(self):
        self.transfer_inst.guest_share_primes.remote(self.vss.prime,
                                                     role="host",
                                                     idx=-1)

    def sync_share_to_host(self):
        for i in range(self.share_amount - 1):
            self.transfer_inst.guest_share_secret.remote(self.secret_sharing[i],
                                                         role="host",
                                                         idx=i)
        self.x_plus_y = self.secret_sharing[-1]

    def recv_share_from_host(self):
        for i in range(self.share_amount - 1):
            self.y_recv.append(self.transfer_inst.host_share_to_guest.get(idx=i))

    def recv_host_sum_from_host(self):
        for i in range(self.share_amount - 1):
            self.host_sum_recv.append(self.transfer_inst.host_sum.get(idx=i))

    def fit(self, data_inst):
        LOGGER.info("begin to make guest data")
        self._init_data(data_inst)

        LOGGER.info("sync primes to host")
        self.sync_primes_to_host()

        LOGGER.info("split data into multiple random parts")
        self.secure()

        LOGGER.info("share one random part data to multiple hosts")
        self.sync_share_to_host()

        LOGGER.info("get share of one random part data from multiple hosts")
        self.recv_share_from_host()

        LOGGER.info("begin to get sum of multiple party")
        self.sharing_sum()

        LOGGER.info("receive host sum from host")
        self.recv_host_sum_from_host()

        self.reconstruct()

        LOGGER.info("success to calculate privacy sum")

        data_output = self.secret_sum

        return data_output


