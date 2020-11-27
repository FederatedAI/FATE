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

import time

from python.fate_arch.session import computing_session
from python.federatedml.util import LOGGER
from python.federatedml.param.secure_sharing_sum_param import SecureSharingSumParam
from python.federatedml.transfer_variable.transfer_class.secret_sharing_sum_transfer_variable import \
    SecretSharingSumTransferVariables
from python.federatedml.util.param_extract import ParamExtract
from python.federatedml.secret_sharing_sum.bash_secret_sharing_sum import BaseSecretSharingSum
from python.federatedml.secret_sharing_sum.secret_sharing_sum_util import Primes


class SecretSharingSumGuest(BaseSecretSharingSum):
    def __init__(self):
        super(SecretSharingSumGuest, self).__init__()
        self.data_num = None
        self.data_set = None
        self.seed = None
        self.transfer_inst = SecretSharingSumTransferVariables()
        self.model_param = SecureSharingSumParam()
        self.primes = Primes()
        self.data_output = None
        self.model_output = None
        self.data = None
        self.host_party_idlist = []

    def _init_runtime_parameters(self, component_parameters):
        param_extracter = ParamExtract()
        param = param_extracter.parse_param_from_config(self.model_param, component_parameters)
        self.role = self.component_properties.parse_component_param(component_parameters, param).role
        self.host_party_idlist = self.component_properties.parse_component_param(component_parameters,
                                                                                 param).host_party_idlist
        self.share_amount = len(self.host_party_idlist) + 1
        self._init_model(param)
        return param

    def _init_model(self, model_param):
        self.need_verify = model_param.need_verify
        self.partition = model_param.partition

    def _init_data(self, arg=None):
        _, _, self.data = self.component_properties.extract_input_data(arg)
        self.data_set = [(k, int(v)) for k, v in self.data['args'].get_all()]
        self.x = computing_session.parallelize(self.data_set, include_key=True, partition=self.partition)
        self.prime = self.primes.get_large_enough_prime(batch=[100000000])

    def sync_primes_to_host(self):
        self.transfer_inst.guest_share_primes.remote(self.prime,
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

    def run(self, component_parameters=None, args=None):
        LOGGER.info("begin to init parameters of multi-party privacy summation guest")
        self._init_runtime_parameters(component_parameters)

        LOGGER.info("begin to make guest data")
        self._init_data(args)

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

        LOGGER.info("success to calculate privacy sum, it is {}".format(self.secret_sum))

    def save_data(self):
        save_result = computing_session.parallelize(self.secret_sum, include_key=True, partition=self.partition)
        date = time.strftime("%Y_%m_%d", time.localtime())
        tablename = 'secret_sharing_sum_'+date

        res = save_result.save_as(name=tablename, namespace='secret_sharing_sum',
                                  partition=self.partition)

        metas = self.data['args'].get_metas()
        res.save_metas(metas)
        return res
