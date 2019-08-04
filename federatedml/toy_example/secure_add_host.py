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

from arch.api import eggroll
from arch.api import federation
from arch.api.utils import log_utils
from federatedml.util.transfer_variable.secure_add_example_transfer_variable import SecureAddExampleTransferVariable
from federatedml.param.secure_add_example_param import SecureAddExampleParam
from federatedml.model_base import ModelBase
import numpy as np


LOGGER = log_utils.getLogger()


class SecureAddHost(ModelBase):
    def __init__(self):
        self.y = None
        self.y1 = None
        self.y2 = None
        self.x2 = None
        self.x2_plus_y2 = None
        self.transfer_inst = SecureAddExampleTransferVariable()
        self.model_param = SecureAddExampleParam()

    def _init_model(self, model_param):
        self.data_num = model_param.data_num
        self.partition = model_param.partition
        self.seed = model_param.seed

    def _init_data(self):
        kvs = [(i, 1) for i in range(self.data_num)]
        self.y = eggroll.parallelize(kvs, include_key=True, partition=self.partition)

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

    def run(self, component_parameters=None, args=None):
        LOGGER.info("begin to init parameters of secure add example host")
        self._init_runtime_parameters(component_parameters)

        LOGGER.info("begin to make host data")
        self._init_data()

        LOGGER.info("split data into two random parts")
        self.secure()

        LOGGER.info("get share of one random part data from guest")
        self.recv_share_from_guest()

        LOGGER.info("share one random part data to guest")
        self.sync_share_to_guest()

        LOGGER.info("begin to get sum of host and guest")
        host_sum = self.add()

        LOGGER.info("send host sum to guest")
        self.sync_host_sum_to_guest(host_sum)


