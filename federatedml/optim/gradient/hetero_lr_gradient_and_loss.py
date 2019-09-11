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
from federatedml.framework.hetero.sync import loss_sync
from federatedml.optim.gradient import hetero_lr_gradient_sync
from federatedml.util.fate_operator import reduce_add

LOGGER = log_utils.getLogger()


class Guest(hetero_lr_gradient_sync.Guest, loss_sync.Guest):
    def __init__(self):
        self.host_forwards = None
        self.half_wx = None
        self.aggregated_wx = None

    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.host_forward_dict,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.guest_gradient,
                                     transfer_variables.guest_optim_gradient)

        self._register_loss_sync(transfer_variables.host_loss_regular,
                                 transfer_variables.loss,
                                 transfer_variables.loss_immediate)

    def compute_gradient_procedure(self, data_instances, encrypted_calculator, lr_weights, optimizer,
                                   n_iter_, batch_index):
        """
        Compute gradients.
        gradient = (1/N)*∑(1/2*ywx-1)*1/2yx = (1/N)*∑(0.25 * wx - 0.5 * y) * x, where y = 1 or -1

        Define wx as guest_forward or host_forward
        Define (0.25 * wx - 0.5 * y) as fore_gradient

        Then, gradient = fore_gradient * x


        Parameters
        ----------
        data_instances: DTable of Instance, input data

        encrypted_calculator: Use for different encrypted methods

        lr_weights: LogisticRegressionWeights
            Stores coef_ and intercept_ of lr

        n_iter_: int, current number of iter.

        batch_index: int, use to obtain current encrypted_calculator
        """
        current_suffix = (n_iter_, batch_index)
        host_forwards = self.get_host_forward(suffix=current_suffix)
        self.host_forwards = host_forwards
        wx = data_instances.mapValues(
            lambda v: np.dot(v.features, lr_weights.coef_) + lr_weights.intercept_)

        self.half_wx = wx
        self.aggregated_wx = encrypted_calculator[batch_index].encrypt(wx)

        for host_forward in host_forwards:
            self.aggregated_wx = self.aggregated_wx.join(host_forward, lambda g, h: g + h)
        fore_gradient = self.aggregated_wx.join(data_instances, lambda wx, d: 0.25 * wx - 0.5 * d.label)

        self.remote_fore_gradient(fore_gradient, suffix=current_suffix)

        unilateral_gradient = self.compute_gradient(data_instances,
                                                    fore_gradient,
                                                    lr_weights.fit_intercept)
        unilateral_gradient = optimizer.add_regular_to_grad(unilateral_gradient, lr_weights)

        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        return optimized_gradient, fore_gradient, host_forwards

    def compute_loss(self, data_instances, n_iter_, batch_index, loss_norm=None):
        """
        Compute hetero-lr loss for:
        loss = (1/N)*∑(log2 - 1/2*ywx + 1/8*(wx)^2), where y is label, w is model weight and x is features
        where (wx)^2 = (Wg * Xg + Wh * Xh)^2 = (Wg*Xg)^2 + (Wh*Xh)^2 + 2 * Wg*Xg * Wh*Xh

        Then loss = log2 - (1/N)*0.5*∑ywx + (1/N)*0.125*[∑(Wg*Xg)^2 + ∑(Wh*Xh)^2 + 2 * ∑(Wg*Xg * Wh*Xh)]

        where Wh*Xh is a table obtain from host and ∑(Wh*Xh)^2 is a sum number get from host.
        """
        current_suffix = (n_iter_, batch_index)
        n = data_instances.count()
        ywx = self.aggregated_wx.join(data_instances, lambda wx, d: wx * int(d.label)).reduce(reduce_add)
        self_wx_square = self.half_wx.mapValues(lambda x: np.square(x)).reduce(reduce_add)

        loss_list = []
        wx_squares = self.get_host_loss_immediate(suffix=current_suffix)

        if loss_norm is not None:
            host_loss_regular = self.get_host_loss_regular(suffix=current_suffix)
        else:
            host_loss_regular = []

        for host_idx, host_forward in enumerate(self.host_forwards):
            wxg_wxh = self.half_wx.join(host_forward, lambda wxg, wxh: wxg * wxh).reduce(reduce_add)
            loss = np.log(2) - 0.5 * (1 / n) * ywx + 0.125 * (1 / n) * \
                                (self_wx_square + wx_squares[host_idx] + 2 * wxg_wxh)
            if loss_norm is not None:
                loss += loss_norm
                loss += host_loss_regular[host_idx]
            loss_list.append(loss)
        self.sync_loss_info(loss_list, suffix=current_suffix)


class Host(hetero_lr_gradient_sync.Host, loss_sync.Host):

    def __init__(self):
        self.half_wx = None

    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.host_forward_dict,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.host_gradient,
                                     transfer_variables.host_optim_gradient)

        self._register_loss_sync(transfer_variables.host_loss_regular,
                                 transfer_variables.loss,
                                 transfer_variables.loss_immediate)

    def compute_gradient_procedure(self, data_instances, lr_weights,
                                   encrypted_calculator, optimizer,
                                   n_iter_, batch_index):
        """
        Compute gradients.
        gradient = (1/N)*∑(1/2*ywx-1)*1/2yx = (1/N)*∑(0.25 * wx - 0.5 * y) * x, where y = 1 or -1

        Define ∑(0.25 * wx - 0.5 * y) as fore_gradient

        Parameters
        ----------
        data_instances: DTable of Instance, input data

        lr_weights: LogisticRegressionWeights
            Stores coef_ and intercept_ of lr

        encrypted_calculator: Use for different encrypted methods

        optimizer: optimizer obj

        n_iter_: int, current iter nums

        batch_index: int, use to obtain current encrypted_calculator

        """
        current_suffix = (n_iter_, batch_index)
        wx = data_instances.mapValues(lambda v: np.dot(v.features, lr_weights.coef_) + lr_weights.intercept_)
        self.half_wx = wx
        host_forward = encrypted_calculator[batch_index].encrypt(wx)
        self.remote_host_forward(host_forward, suffix=current_suffix)

        fore_gradient = self.get_fore_gradient(suffix=current_suffix)

        unilateral_gradient = self.compute_gradient(data_instances,
                                                    fore_gradient,
                                                    lr_weights.fit_intercept)
        unilateral_gradient = optimizer.add_regular_to_grad(unilateral_gradient, lr_weights)

        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        return optimized_gradient, fore_gradient

    def compute_loss(self, lr_weights, optimizer, n_iter_, batch_index):
        """
        Compute hetero-lr loss for:
        loss = (1/N)*∑(log2 - 1/2*ywx + 1/8*(wx)^2), where y is label, w is model weight and x is features
        where (wx)^2 = (Wg * Xg + Wh * Xh)^2 = (Wg*Xg)^2 + (Wh*Xh)^2 + 2 * Wg*Xg * Wh*Xh

        Then loss = log2 - (1/N)*0.5*∑ywx + (1/N)*0.125*[∑(Wg*Xg)^2 + ∑(Wh*Xh)^2 + 2 * ∑(Wg*Xg * Wh*Xh)]

        where Wh*Xh is a table obtain from host and ∑(Wh*Xh)^2 is a sum number get from host.
        """
        current_suffix = (n_iter_, batch_index)
        self_wx_square = self.half_wx.mapValues(lambda x: np.square(x)).reduce(reduce_add)
        self.remote_loss_immediate(self_wx_square, suffix=current_suffix)

        loss_regular = optimizer.loss_norm(lr_weights.coef_)
        self.remote_loss_regular(loss_regular, suffix=current_suffix)


class Arbiter(hetero_lr_gradient_sync.Arbiter, loss_sync.Arbiter):
    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.guest_gradient,
                                     transfer_variables.host_gradient,
                                     transfer_variables.guest_optim_gradient,
                                     transfer_variables.host_optim_gradient)
        self._register_loss_sync(transfer_variables.loss)

    def compute_gradient_procedure(self, cipher_operator, optimizer, n_iter_, batch_index):
        """
        Compute gradients.
        gradient = (1/N)*∑(1/2*ywx-1)*1/2yx = (1/N)*∑(0.25 * wx - 0.5 * y) * x, where y = 1 or -1

        Received local_gradients from guest and hosts. Merge and optimize, then separate and remote back.

        Parameters
        ----------
        cipher_operator: Use for encryption

        optimizer: optimizer that get delta gradient of this iter

        n_iter_: int, current iter nums

        batch_index: int, use to obtain current encrypted_calculator

        """
        current_suffix = (n_iter_, batch_index)

        host_gradients, guest_gradient = self.get_local_gradient(current_suffix)

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

        self.remote_local_gradient(host_optim_gradients, guest_optim_gradient, current_suffix)
        return delta_grad

    def compute_loss(self, cipher, n_iter_, batch_index):
        """
        Compute hetero-lr loss for:
        loss = (1/N)*∑(log2 - 1/2*ywx + 1/8*(wx)^2), where y is label, w is model weight and x is features
        where (wx)^2 = (Wg * Xg + Wh * Xh)^2 = (Wg*Xg)^2 + (Wh*Xh)^2 + 2 * Wg*Xg * Wh*Xh

        Then loss = log2 - (1/N)*0.5*∑ywx + (1/N)*0.125*[∑(Wg*Xg)^2 + ∑(Wh*Xh)^2 + 2 * ∑(Wg*Xg * Wh*Xh)]

        where Wh*Xh is a table obtain from host and ∑(Wh*Xh)^2 is a sum number get from host.
        """
        current_suffix = (n_iter_, batch_index)
        loss_list = self.sync_loss_info(suffix=current_suffix)
        de_loss_list = cipher.decrypt_list(loss_list)
        return de_loss_list
