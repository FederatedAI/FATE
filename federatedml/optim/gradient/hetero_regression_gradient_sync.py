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

import numpy as np

from arch.api.utils import log_utils
from federatedml.framework.hetero.sync import gradient_sync
from federatedml.optim.gradient.hetero_regression_gradient import HeteroGradientComputer
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class Guest(gradient_sync.Guest, HeteroGradientComputer):
    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                guest_gradient_transfer, guest_optim_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = guest_gradient_transfer
        self.unilateral_optim_gradient_transfer = guest_optim_gradient_transfer

    def get_host_forward(self, suffix=tuple()):
        host_forward = self.host_forward_transfer.get(idx=-1, suffix=suffix)
        return host_forward

    def remote_fore_gradient(self, fore_gradient, suffix=tuple()):
        self.fore_gradient_transfer.remote(obj=fore_gradient, role=consts.HOST, idx=-1, suffix=suffix)

    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)
        return optimized_gradient


class Host(gradient_sync.Host, HeteroGradientComputer):
    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                host_gradient_transfer, host_optim_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = host_gradient_transfer
        self.unilateral_optim_gradient_transfer = host_optim_gradient_transfer

    def remote_host_forward(self, host_forward, suffix=tuple()):
        self.host_forward_transfer.remote(obj=host_forward, role=consts.GUEST, idx=0, suffix=suffix)

    def get_fore_gradient(self, suffix=tuple()):
        host_forward = self.fore_gradient_transfer.get(idx=0, suffix=suffix)
        return host_forward

    def compute_intermediate(self):
        pass

    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)
        return optimized_gradient


class Arbiter(gradient_sync.Arbiter):
    def _register_gradient_sync(self, guest_gradient_transfer, host_gradient_transfer,
                                guest_optim_gradient_transfer, host_optim_gradient_transfer):
        self.guest_gradient_transfer = guest_gradient_transfer
        self.host_gradient_transfer = host_gradient_transfer
        self.guest_optim_gradient_transfer = guest_optim_gradient_transfer
        self.host_optim_gradient_transfer = host_optim_gradient_transfer

    def get_local_gradient(self, suffix=tuple()):
        host_gradients = self.host_gradient_transfer.get(idx=-1, suffix=suffix)
        LOGGER.info("Get host_gradient from Host")

        guest_gradient = self.guest_gradient_transfer.get(idx=0, suffix=suffix)
        LOGGER.info("Get guest_gradient from Guest")
        return host_gradients, guest_gradient

    def remote_local_gradient(self, host_optim_gradients, guest_optim_gradient, suffix=tuple()):
        for idx, host_optim_gradient in enumerate(host_optim_gradients):
            self.host_optim_gradient_transfer.remote(host_optim_gradient,
                                                     role=consts.HOST,
                                                     idx=idx,
                                                     suffix=suffix)

        self.guest_optim_gradient_transfer.remote(guest_optim_gradient,
                                                  role=consts.GUEST,
                                                  idx=0,
                                                  suffix=suffix)
