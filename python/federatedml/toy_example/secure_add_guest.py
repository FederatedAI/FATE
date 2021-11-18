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
from fate_arch.federation import Tag, get, remote
from fate_arch.session import PartiesInfo as Parties
from fate_arch.session import computing_session as session
from federatedml.model_base import ComponentOutput, ModelBase
from federatedml.param.secure_add_example_param import SecureAddExampleParam
from federatedml.transfer_variable.transfer_class.secure_add_example_transfer_variable import (
    SecureAddExampleTransferVariable,
)
from federatedml.util import LOGGER


class SecureAddGuest(ModelBase):
    def __init__(self):
        super(SecureAddGuest, self).__init__()
        self.x = None
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.x1_plus_y1 = None
        self.data_num = None
        self.partition = None
        self.seed = None
        self.transfer_inst = SecureAddExampleTransferVariable()
        self.model_param = SecureAddExampleParam()
        self.data_output = None
        self.model_output = None

    def _init_runtime_parameters(self, cpn_input):
        self.model_param.update(cpn_input.parameters)
        self._init_model()

    def _init_model(self):
        self.data_num = self.model_param.data_num
        self.partition = self.model_param.partition
        self.seed = self.model_param.seed

    def _init_data(self):
        kvs = [(i, 1) for i in range(self.data_num)]
        self.x = session.parallelize(kvs, include_key=True, partition=self.partition)

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
        print("host sum is %.4f" % host_sum)
        print("guest sum is %.4f" % guest_sum)
        secure_sum = host_sum + guest_sum

        print("Secure Add Result is %.4f" % secure_sum)

        return secure_sum

    @Tag("toy")
    def run(self, cpn_input):
        LOGGER.info("begin to init parameters of secure add example guest")

        self._init_runtime_parameters(cpn_input)

        LOGGER.info("begin to make guest data")
        self._init_data()

        LOGGER.info("split data into two random parts")
        self.secure()

        # demo for switch tag context
        with Tag("share"):
            LOGGER.info("share one random part data to host")
            remote(parties=Parties.Host[0], guest_share=self.x2)

        LOGGER.info("get share of one random part data from host")
        self.y1 = get(Parties.Host[0], name="host_share")

        LOGGER.info("begin to get sum of guest and host")
        guest_sum = self.add()

        LOGGER.info("receive host sum from guest")
        host_sum = get(Parties.Host[0], "host_sum")

        secure_sum = self.reconstruct(guest_sum, host_sum)

        assert np.abs(secure_sum - self.data_num * 2) < 1e-6

        LOGGER.info("success to calculate secure_sum, it is {}".format(secure_sum))

        return ComponentOutput(self.save_data(), self.export_model(), self.save_cache())
