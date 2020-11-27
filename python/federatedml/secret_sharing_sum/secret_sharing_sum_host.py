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

import numpy as np

from python.fate_arch.session import computing_session
from python.federatedml.util import LOGGER
from python.federatedml.param.secure_sharing_sum_param import SecureSharingSumParam
from python.federatedml.transfer_variable.transfer_class.secret_sharing_sum_transfer_variable import \
    SecretSharingSumTransferVariables
from python.federatedml.util.param_extract import ParamExtract
from python.federatedml.secret_sharing_sum.bash_secret_sharing_sum import BaseSecretSharingSum


class SecretSharingSumHost(BaseSecretSharingSum):
    def __init__(self):
        super(SecretSharingSumHost, self).__init__()
        self.transfer_inst = SecretSharingSumTransferVariables()
        self.model_param = SecureSharingSumParam()
        self.data_output = None
        self.model_output = None
        self.host_party_idlist = []
        self.local_partyid = -1

    def _init_runtime_parameters(self, component_parameters):
        param_extracter = ParamExtract()
        param = param_extracter.parse_param_from_config(self.model_param, component_parameters)
        self.role = self.component_properties.parse_component_param(component_parameters, param).role
        self.local_partyid = self.component_properties.parse_component_param(component_parameters, param).local_partyid
        self.host_party_idlist = self.component_properties.parse_component_param(component_parameters,
                                                                                 param).host_party_idlist
        self.share_amount = len(self.host_party_idlist)+1
        self._init_model(param)
        return param

    def _init_model(self, model_param):
        self.need_verify = model_param.need_verify
        self.partition = model_param.partition

    def _init_data(self, arg=None):
        _, _, data = self.component_properties.extract_input_data(arg)
        self.data_set = [(k, int(v)) for k, v in data['args'].get_all()]
        self.x = computing_session.parallelize(self.data_set, include_key=True, partition=self.partition)

    def recv_primes_from_guest(self):
        self.prime = self.transfer_inst.guest_share_primes.get(idx=0)

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

    def run(self, component_parameters=None, args=None):
        LOGGER.info("begin to init parameters of multi-party privacy summation host")
        self._init_runtime_parameters(component_parameters)

        LOGGER.info("begin to make host data")
        self._init_data(args)

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
