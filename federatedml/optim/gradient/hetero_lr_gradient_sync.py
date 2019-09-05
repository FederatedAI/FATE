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
        self.guest_gradient_transfer = guest_gradient_transfer
        self.guest_optim_gradient_transfer = guest_optim_gradient_transfer

    def compute_intermediate(self, data_instances, lr_variables, compute_wx, encrypted_calculator, batch_index):
        """
        Compute W * X + b and (W * X + b)^2, where X is the input data, W is the coefficient of lr,
        and b is the interception
        Parameters
        ----------
        data_instances: DTable of Instance, input data

        lr_variables: LogisticRegressionVariables
            Stores coef_ and intercept_ of lr

        compute_wx: Function Type, a compute_wx func

        encrypted_calculator: Use for different encrypted methods

        batch_index: int, use to obtain current encrypted_calculator

        """
        wx = compute_wx(data_instances, lr_variables.coef_, lr_variables.intercept_)

        en_wx = encrypted_calculator[batch_index].encrypt(wx)
        wx_square = wx.mapValues(lambda v: np.square(v))
        en_wx_square = encrypted_calculator[batch_index].encrypt(wx_square)

        en_wx_join_en_wx_square = en_wx.join(en_wx_square, lambda wx, wx_square: (wx, wx_square))
        guest_forward = en_wx_join_en_wx_square.join(wx, lambda e, wx: (e[0], e[1], wx))
        return guest_forward

    def aggregate_host_result(self, host_forward, guest_forward):
        """
        Compute (en_wx_g + en_wx_h)^2 = en_wx_g^2 + en_wx_h^2 + 2 * wx_g * en_wx_h ,
         where en_wx_g is the encrypted W * X + b of guest, wx_g is unencrypted W * X + b,
        and en_wx_h is the encrypted W * X + b of host.
        Parameters
        ----------
        host_forward: DTable, include encrypted W * X and (W * X)^2

        guest_forward: DTable, include encrypted W * X + b, (W * X + b)^2 and unencrypted wx


        Returns
        ----------
        aggregate_forward_res
        list
            include W * X and (W * X)^2 federate with guest and host
        """
        aggregate_forward_res = guest_forward.join(host_forward,
                                                   lambda g, h: (g[0] + h[0], g[1] + h[1] + 2 * g[2] * h[0]))

        en_aggregate_wx = aggregate_forward_res.mapValues(lambda v: v[0])
        en_aggregate_wx_square = aggregate_forward_res.mapValues(lambda v: v[1])

        # self.rubbish_bin.append(aggregate_forward_res)
        return en_aggregate_wx, en_aggregate_wx_square

    def compute_gradient_procedure(self, data_instances, lr_variables,
                                   compute_wx, encrypted_calculator,
                                   n_iter_, batch_index):
        current_suffix = (n_iter_, batch_index)

        guest_forward = self.compute_intermediate(data_instances, lr_variables,
                                                  compute_wx, encrypted_calculator, batch_index)
        host_forward = self.host_forward_dict_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get host_forward from host")

        en_aggregate_wx, en_aggregate_wx_square = self.aggregate_host_result(host_forward, guest_forward)

        fore_gradient = self.compute_fore_gradient(data_instances, en_aggregate_wx)
        self.fore_gradient_transfer.remote(fore_gradient, role=consts.HOST, idx=0, suffix=current_suffix)
        LOGGER.info("Remote fore_gradient to Host")

        guest_gradient, loss = self.compute_gradient_and_loss(data_instances,
                                                              fore_gradient,
                                                              en_aggregate_wx,
                                                              en_aggregate_wx_square,
                                                              lr_variables.fit_intercept)

        self.guest_gradient_transfer.remote(guest_gradient, role=consts.ARBITER, idx=0, suffix=current_suffix)
        LOGGER.info("Remote guest_gradient to arbiter")

        optim_guest_gradient = self.guest_optim_gradient_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get optim_guest_gradient from arbiter")
        return optim_guest_gradient, loss, fore_gradient


class Host(gradient_sync.Host, HeteroLogisticGradientComputer):
    def _register_gradient_sync(self, host_forward_dict_transfer, fore_gradient_transfer,
                                host_gradient_transfer, host_optim_gradient_transfer):
        self.host_forward_dict_transfer = host_forward_dict_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.host_gradient_transfer = host_gradient_transfer
        self.host_optim_gradient_transfer = host_optim_gradient_transfer

    def compute_intermediate(self, data_instances, lr_variables, compute_wx, encrypted_calculator, batch_index):
        """
        Compute W * X + b and (W * X + b)^2, where X is the input data, W is the coefficient of lr,
        and b is the interception
        Parameters
        ----------
        data_instances: DTable of Instance, input data

        lr_variables: LogisticRegressionVariables
            Stores coef_ and intercept_ of lr

        compute_wx: Function Type, a compute_wx func

        encrypted_calculator: Use for different encrypted methods

        batch_index: int, use to obtain current encrypted_calculator

        """
        wx = compute_wx(data_instances, lr_variables.coef_, lr_variables.intercept_)

        en_wx = encrypted_calculator[batch_index].encrypt(wx)
        wx_square = wx.mapValues(lambda v: np.square(v))
        en_wx_square = encrypted_calculator[batch_index].encrypt(wx_square)
        host_forward = en_wx.join(en_wx_square, lambda wx, wx_square: (wx, wx_square))

        return host_forward

    def compute_gradient_procedure(self, data_instances, lr_variables,
                                   compute_wx, encrypted_calculator,
                                   n_iter_, batch_index):
        current_suffix = (n_iter_, batch_index)

        host_forward = self.compute_intermediate(data_instances, lr_variables, compute_wx,
                                                 encrypted_calculator, batch_index)

        self.host_forward_dict_transfer.remote(host_forward, role=consts.GUEST, idx=0, suffix=current_suffix)
        LOGGER.info("Remote host_forward to guest")

        fore_gradient = self.fore_gradient_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get fore_gradient from guest")

        host_gradient = self.compute_gradient(data_instances,
                                              fore_gradient,
                                              fit_intercept=False)

        self.host_gradient_transfer.remote(host_gradient, role=consts.ARBITER, idx=0, suffix=current_suffix)
        LOGGER.info("Remote host_gradient to arbiter")

        optim_host_gradient = self.host_optim_gradient_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get optim_guest_gradient from arbiter")

        return optim_host_gradient, fore_gradient


class Arbiter(gradient_sync.Arbiter):
    def _register_gradient_sync(self, guest_gradient_transfer, host_gradient_transfer,
                                guest_optim_gradient_transfer, host_optim_gradient_transfer):
        self.guest_gradient_transfer = guest_gradient_transfer
        self.host_gradient_transfer = host_gradient_transfer
        self.guest_optim_gradient_transfer = guest_optim_gradient_transfer
        self.host_optim_gradient_transfer = host_optim_gradient_transfer

    def compute_gradient_procedure(self, cipher_operator, optimizer, n_iter_, batch_index):
        current_suffix = (n_iter_, batch_index)

        host_gradient = self.host_gradient_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get host_gradient from Host")

        guest_gradient = self.guest_gradient_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get guest_gradient from Guest")
        host_gradient, guest_gradient = np.array(host_gradient), np.array(guest_gradient)
        gradient = np.hstack((host_gradient, guest_gradient))

        grad = np.array(cipher_operator.decrypt_list(gradient))
        delta_grad = optimizer.apply_gradients(grad)
        separate_optim_gradient = self.separate(delta_grad,
                                                [host_gradient.shape[0],
                                                 guest_gradient.shape[0]])
        host_optim_gradient = separate_optim_gradient[0]
        guest_optim_gradient = separate_optim_gradient[1]

        self.host_optim_gradient_transfer.remote(host_optim_gradient,
                                                 role=consts.HOST,
                                                 idx=0,
                                                 suffix=current_suffix)

        self.guest_optim_gradient_transfer.remote(guest_optim_gradient,
                                                  role=consts.GUEST,
                                                  idx=0,
                                                  suffix=current_suffix)
