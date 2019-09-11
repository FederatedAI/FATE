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
from federatedml.optim.gradient.poisson_gradient import HeteroPoissonGradientComputer
from federatedml.util import consts
from operator import add

LOGGER = log_utils.getLogger()

class Guest(gradient_sync.Guest, HeteroPoissonGradientComputer):
    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                guest_gradient_transfer, optim_guest_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.guest_gradient_transfer = guest_gradient_transfer
        self.optim_guest_gradient_transfer = optim_guest_gradient_transfer

    def compute_intermediate(self, data_instances, model_variables, exposure):
        """
        Compute W * X + b and exp(W * X + b), where X is the input data, W is the coefficient of model,
        and b is the interception
        Parameters
        ----------
        data_instances: DTable of Instance, input data

        model_variables: PoissonRegressionVariables
            Stores coef_ and intercept_ of model

        encrypted_calculator: Use for different encrypted methods

        batch_index: int, use to obtain current encrypted_calculator

        exposure: DTable of exposure for rate prediction
        """
        guest_forward = self.compute_wx_mu(data_instances, model_variables.coef_, model_variables.intercept_,
                                           exposure)
        return guest_forward

    def compute_gradient_procedure(self, data_instances, model_variables,
                                   encrypted_calculator,
                                   n_iter_, batch_index, exposure):
        current_suffix = (n_iter_, batch_index)

        guest_forward = self.compute_intermediate(data_instances, model_variables, exposure)

        host_forward = self.host_forward_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get host_forward from host")

        fore_gradient = self.compute_fore_gradient(guest_forward, host_forward)
        self.fore_gradient_transfer.remote(fore_gradient, role=consts.HOST, idx=0, suffix=current_suffix)
        LOGGER.info("Remote fore_gradient to Host")

        guest_gradient = self.compute_gradient(data_instances, fore_gradient,
                                               model_variables.fit_intercept)
        loss = self.compute_loss(guest_forward, host_forward, exposure)

        self.guest_gradient_transfer.remote(guest_gradient, role=consts.ARBITER, idx=0, suffix=current_suffix)
        LOGGER.info("Remote guest_gradient to arbiter")

        optim_guest_gradient = self.optim_guest_gradient_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get optim_guest_gradient from arbiter")
        return optim_guest_gradient, loss


class Host(gradient_sync.Host, HeteroPoissonGradientComputer):
    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                host_gradient_transfer, optim_host_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.host_gradient_transfer = host_gradient_transfer
        self.optim_host_gradient_transfer = optim_host_gradient_transfer

    def compute_intermediate(self, data_instances, model_variables, encrypted_calculator, batch_index):
        """
        Compute W * X + b and exp(W * X + b), where X is the input data, W is the coefficient of model,
        and b is the interception
        Parameters
        ----------
        data_instances: DTable of Instance, input data

        model_variables: PoissonRegressionVariables
            Stores coef_ and intercept_ of model

        encrypted_calculator: Use for different encrypted methods

        batch_index: int, use to obtain current encrypted_calculator

        """
        mu = self.compute_mu(data_instances, model_variables.coef_, model_variables.intercept_)
        en_mu = encrypted_calculator[batch_index].encrypt(mu)
        wx = self.compute_wx(data_instances, model_variables.coef_, model_variables.intercept_)
        en_wx = encrypted_calculator[batch_index].encrypt(wx)
        host_forward = en_wx.join(en_mu, lambda wx, mu: (wx, mu))
        return host_forward

    def compute_gradient_procedure(self, data_instances, model_variables, encrypted_calculator,
                                   n_iter_, batch_index):
        current_suffix = (n_iter_, batch_index)

        host_forward = self.compute_intermediate(data_instances, model_variables,
                                                 encrypted_calculator, batch_index)

        self.host_forward_transfer.remote(host_forward, role=consts.GUEST, idx=0, suffix=current_suffix)
        LOGGER.info("Remote host_forward to guest")

        fore_gradient = self.fore_gradient_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get fore_gradient from guest")

        host_gradient = self.compute_gradient(data_instances,
                                              fore_gradient,
                                              fit_intercept=False)

        self.host_gradient_transfer.remote(host_gradient, role=consts.ARBITER, idx=0, suffix=current_suffix)
        LOGGER.info("Remote host_gradient to arbiter")

        optim_host_gradient = self.optim_host_gradient_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get optim_guest_gradient from arbiter")

        return optim_host_gradient


class Arbiter(gradient_sync.Arbiter):
    def _register_gradient_sync(self, guest_gradient_transfer, host_gradient_transfer,
                                optim_guest_gradient_transfer, optim_host_gradient_transfer):
        self.guest_gradient_transfer = guest_gradient_transfer
        self.host_gradient_transfer = host_gradient_transfer
        self.optim_guest_gradient_transfer = optim_guest_gradient_transfer
        self.optim_host_gradient_transfer = optim_host_gradient_transfer

    def compute_gradient_procedure(self, cipher_operator, optimizer, n_iter_, batch_index):
        current_suffix = (n_iter_, batch_index)

        host_gradient = self.host_gradient_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get host_gradient from Host")

        guest_gradient = self.guest_gradient_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get guest_gradient from Guest")
        host_gradient, guest_gradient = np.array(host_gradient), np.array(guest_gradient)
        gradient = np.hstack((host_gradient, guest_gradient))

        grad = cipher_operator.decrypt_list(gradient)
        delta_grad = optimizer.apply_gradients(grad)
        separate_optim_gradient = self.separate(delta_grad,
                                                [host_gradient.shape[0],
                                                 guest_gradient.shape[0]])
        optim_host_gradient = separate_optim_gradient[0]
        optim_guest_gradient = separate_optim_gradient[1]

        self.optim_host_gradient_transfer.remote(optim_host_gradient,
                                                 role=consts.HOST,
                                                 idx=0,
                                                 suffix=current_suffix)

        self.optim_guest_gradient_transfer.remote(optim_guest_gradient,
                                                  role=consts.GUEST,
                                                  idx=0,
                                                  suffix=current_suffix)
