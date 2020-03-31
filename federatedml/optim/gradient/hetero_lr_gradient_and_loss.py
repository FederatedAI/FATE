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
from federatedml.optim.gradient import hetero_linear_model_gradient
from federatedml.util.fate_operator import reduce_add, vec_dot

LOGGER = log_utils.getLogger()


class Guest(hetero_linear_model_gradient.Guest, loss_sync.Guest):
    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.host_forward_dict,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.guest_gradient,
                                     transfer_variables.guest_optim_gradient)

        self._register_loss_sync(transfer_variables.host_loss_regular,
                                 transfer_variables.loss,
                                 transfer_variables.loss_intermediate)

    def compute_and_aggregate_forwards(self, data_instances, model_weights, encrypted_calculator, batch_index,
                                       offset=None):
        """
        gradient = (1/N)*∑(1/2*ywx-1)*1/2yx = (1/N)*∑(0.25 * wx - 0.5 * y) * x, where y = 1 or -1
        Define wx as guest_forward or host_forward
        Define (0.25 * wx - 0.5 * y) as fore_gradient

        """

        half_wx = data_instances.mapValues(
            lambda v: vec_dot(v.features, model_weights.coef_) + model_weights.intercept_)
        self.forwards = half_wx
        # LOGGER.debug("half_wx: {}".format(half_wx.take(20)))
        self.aggregated_forwards = encrypted_calculator[batch_index].encrypt(half_wx)

        for host_forward in self.host_forwards:
            self.aggregated_forwards = self.aggregated_forwards.join(host_forward, lambda g, h: g + h)
        fore_gradient = self.aggregated_forwards.join(data_instances, lambda wx, d: 0.25 * wx - 0.5 * d.label)
        return fore_gradient

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
        ywx = self.aggregated_forwards.join(data_instances, lambda wx, d: wx * int(d.label)).reduce(reduce_add)
        self_wx_square = self.forwards.mapValues(lambda x: np.square(x)).reduce(reduce_add)

        loss_list = []
        wx_squares = self.get_host_loss_intermediate(suffix=current_suffix)

        if loss_norm is not None:
            host_loss_regular = self.get_host_loss_regular(suffix=current_suffix)
        else:
            host_loss_regular = []

        # for host_idx, host_forward in enumerate(self.host_forwards):
        if len(self.host_forwards) > 1:
            LOGGER.info("More than one host exist, loss is not available")
        else:
            host_forward = self.host_forwards[0]
            wx_square = wx_squares[0]
            wxg_wxh = self.forwards.join(host_forward, lambda wxg, wxh: wxg * wxh).reduce(reduce_add)
            loss = np.log(2) - 0.5 * (1 / n) * ywx + 0.125 * (1 / n) * \
                                                     (self_wx_square + wx_square + 2 * wxg_wxh)
            if loss_norm is not None:
                loss += loss_norm
                loss += host_loss_regular[0]
            loss_list.append(loss)
        LOGGER.debug("In compute_loss, loss list are: {}".format(loss_list))
        self.sync_loss_info(loss_list, suffix=current_suffix)

    def compute_forward_hess(self, data_instances, delta_s, host_forwards):
        """
        To compute Hessian matrix, y, s are needed.
        g = (1/N)*∑(0.25 * wx - 0.5 * y) * x
        y = ∇2^F(w_t)s_t = g' * s = (1/N)*∑(0.25 * x * s) * x
        define forward_hess = (1/N)*∑(0.25 * x * s)
        """
        forwards = data_instances.mapValues(
            lambda v: (vec_dot(v.features, delta_s.coef_) + delta_s.intercept_) * 0.25)
        for host_forward in host_forwards:
            forwards = forwards.join(host_forward, lambda g, h: g + (h * 0.25))
        # forward_hess = forwards.mapValues(lambda x: 0.25 * x / sample_size)
        hess_vector = hetero_linear_model_gradient.compute_gradient(data_instances,
                                                                    forwards,
                                                                    delta_s.fit_intercept)
        return forwards, np.array(hess_vector)


class Host(hetero_linear_model_gradient.Host, loss_sync.Host):
    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.host_forward_dict,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.host_gradient,
                                     transfer_variables.host_optim_gradient)

        self._register_loss_sync(transfer_variables.host_loss_regular,
                                 transfer_variables.loss,
                                 transfer_variables.loss_intermediate)

    def compute_forwards(self, data_instances, model_weights):
        """
        forwards = wx
        """
        wx = data_instances.mapValues(lambda v: vec_dot(v.features, model_weights.coef_) + model_weights.intercept_)
        return wx

    def compute_loss(self, lr_weights, optimizer, n_iter_, batch_index, cipher_operator):
        """
        Compute hetero-lr loss for:
        loss = (1/N)*∑(log2 - 1/2*ywx + 1/8*(wx)^2), where y is label, w is model weight and x is features
        where (wx)^2 = (Wg * Xg + Wh * Xh)^2 = (Wg*Xg)^2 + (Wh*Xh)^2 + 2 * Wg*Xg * Wh*Xh

        Then loss = log2 - (1/N)*0.5*∑ywx + (1/N)*0.125*[∑(Wg*Xg)^2 + ∑(Wh*Xh)^2 + 2 * ∑(Wg*Xg * Wh*Xh)]

        where Wh*Xh is a table obtain from host and ∑(Wh*Xh)^2 is a sum number get from host.
        """
        current_suffix = (n_iter_, batch_index)
        self_wx_square = self.forwards.mapValues(lambda x: np.square(x)).reduce(reduce_add)
        en_wx_square = cipher_operator.encrypt(self_wx_square)
        self.remote_loss_intermediate(en_wx_square, suffix=current_suffix)

        loss_regular = optimizer.loss_norm(lr_weights)
        if loss_regular is not None:
            loss_regular = cipher_operator.encrypt(loss_regular)
        self.remote_loss_regular(loss_regular, suffix=current_suffix)


class Arbiter(hetero_linear_model_gradient.Arbiter, loss_sync.Arbiter):
    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.guest_gradient,
                                     transfer_variables.host_gradient,
                                     transfer_variables.guest_optim_gradient,
                                     transfer_variables.host_optim_gradient)
        self._register_loss_sync(transfer_variables.loss)

    def compute_loss(self, cipher, n_iter_, batch_index):
        """
        Compute hetero-lr loss for:
        loss = (1/N)*∑(log2 - 1/2*ywx + 1/8*(wx)^2), where y is label, w is model weight and x is features
        where (wx)^2 = (Wg * Xg + Wh * Xh)^2 = (Wg*Xg)^2 + (Wh*Xh)^2 + 2 * Wg*Xg * Wh*Xh

        Then loss = log2 - (1/N)*0.5*∑ywx + (1/N)*0.125*[∑(Wg*Xg)^2 + ∑(Wh*Xh)^2 + 2 * ∑(Wg*Xg * Wh*Xh)]

        where Wh*Xh is a table obtain from host and ∑(Wh*Xh)^2 is a sum number get from host.
        """
        if self.has_multiple_hosts:
            LOGGER.info("Has more than one host, loss is not available")
            return []

        current_suffix = (n_iter_, batch_index)
        loss_list = self.sync_loss_info(suffix=current_suffix)
        de_loss_list = cipher.decrypt_list(loss_list)
        return de_loss_list

