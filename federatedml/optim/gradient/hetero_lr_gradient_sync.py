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
from federatedml.optim.gradient.logistic_gradient import HeteroLogisticGradientComputer
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class Guest(gradient_sync.Guest, HeteroLogisticGradientComputer):
    def _register_gradient_sync(self, host_forward_dict_transfer, fore_gradient_transfer,
                                guest_gradient_transfer, guest_optim_gradient_transfer):
        self.host_forward_dict_transfer = host_forward_dict_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = guest_gradient_transfer
        self.unilateral_optim_gradient_transfer = guest_optim_gradient_transfer

    def get_host_forward(self, suffix=tuple()):
        host_forward = self.host_forward_dict_transfer.get(idx=-1, suffix=suffix)
        return host_forward

    def remote_fore_gradient(self, fore_gradient, suffix=tuple()):
        self.fore_gradient_transfer.remote(object=fore_gradient, role=consts.HOST, idx=-1, suffix=suffix)

    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)
        return optimized_gradient


class Host(gradient_sync.Host, HeteroLogisticGradientComputer):
    def _register_gradient_sync(self, host_forward_dict_transfer, fore_gradient_transfer,
                                host_gradient_transfer, host_optim_gradient_transfer):
        self.host_forward_dict_transfer = host_forward_dict_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = host_gradient_transfer
        self.unilateral_optim_gradient_transfer = host_optim_gradient_transfer

    def remote_host_forward(self, host_forward, suffix=tuple()):
        self.host_forward_dict_transfer.remote(object=host_forward, role=consts.GUEST, idx=0, suffix=suffix)

    def get_fore_gradient(self, suffix=tuple()):
        host_forward = self.fore_gradient_transfer.get(idx=0, suffix=suffix)
        return host_forward

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

    def update_gradient(self, cipher_operator, optimizer, n_iter_, batch_index):
        current_suffix = (n_iter_, batch_index)

        host_gradients = self.host_gradient_transfer.get(idx=-1, suffix=current_suffix)
        LOGGER.info("Get host_gradient from Host")

        guest_gradient = self.guest_gradient_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get guest_gradient from Guest")

        host_gradients = [np.array(h) for h in host_gradients]
        guest_gradient = np.array(guest_gradient)

        size_list = [h_g.shape[0] for h_g in host_gradients]
        size_list.append(guest_gradient.shape[0])

        gradient = np.hstack((h for h in host_gradients))
        gradient = np.hstack((gradient, guest_gradient))

        grad = np.array(cipher_operator.decrypt_list(gradient))
        delta_grad = optimizer.apply_gradients(grad)
        separate_optim_gradient = self.separate(delta_grad, size_list)
        host_optim_gradients = separate_optim_gradient[: -1]
        guest_optim_gradient = separate_optim_gradient[-1]

        for idx, host_optim_gradient in enumerate(host_optim_gradients):
            self.host_optim_gradient_transfer.remote(host_optim_gradient,
                                                     role=consts.HOST,
                                                     idx=idx,
                                                     suffix=current_suffix)

        self.guest_optim_gradient_transfer.remote(guest_optim_gradient,
                                                  role=consts.GUEST,
                                                  idx=0,
                                                  suffix=current_suffix)

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
