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

from federatedml.framework.hetero.sync import loss_sync
from federatedml.optim.gradient import hetero_linear_model_gradient
from federatedml.util import LOGGER
from federatedml.util.fate_operator import reduce_add, vec_dot


class Guest(hetero_linear_model_gradient.Guest, loss_sync.Guest):

    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.host_forward,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.guest_gradient,
                                     transfer_variables.guest_optim_gradient)

        self._register_loss_sync(transfer_variables.host_loss_regular,
                                 transfer_variables.loss,
                                 transfer_variables.loss_intermediate)

    def compute_half_d(self, data_instances, w, cipher, batch_index, current_suffix):
        if self.use_sample_weight:
            self.half_d = data_instances.mapValues(
                lambda v: (vec_dot(v.features, w.coef_) + w.intercept_ - v.label) * v.weight)
        else:
            self.half_d = data_instances.mapValues(
                lambda v: vec_dot(v.features, w.coef_) + w.intercept_ - v.label)
        return self.half_d

    def compute_and_aggregate_forwards(self, data_instances, half_g, encrypted_half_g, batch_index,
                                       current_suffix, offset=None):
        """
        gradient = (1/N)*sum(wx - y) * x
        Define wx -y  as guest_forward and wx as host_forward
        """
        self.host_forwards = self.get_host_forward(suffix=current_suffix)
        return self.host_forwards

    def compute_loss(self, data_instances, n_iter_, batch_index, loss_norm=None):
        '''
        Compute hetero linr loss:
            loss = (1/N)*\\sum(wx-y)^2 where y is label, w is model weight and x is features
        log(wx - y)^2 = (wx_h)^2 + (wx_g - y)^2 + 2*(wx_h + wx_g - y)
        '''
        current_suffix = (n_iter_, batch_index)
        n = data_instances.count()
        loss_list = []
        host_wx_squares = self.get_host_loss_intermediate(current_suffix)

        if loss_norm is not None:
            host_loss_regular = self.get_host_loss_regular(suffix=current_suffix)
        else:
            host_loss_regular = []
        if len(self.host_forwards) > 1:
            LOGGER.info("More than one host exist, loss is not available")
        else:
            host_forward = self.host_forwards[0]
            host_wx_square = host_wx_squares[0]

            wxy_square = self.half_d.mapValues(lambda x: np.square(x)).reduce(reduce_add)

            loss_gh = self.half_d.join(host_forward, lambda g, h: g * h).reduce(reduce_add)
            loss = (wxy_square + host_wx_square + 2 * loss_gh) / (2 * n)
            if loss_norm is not None:
                loss = loss + loss_norm + host_loss_regular[0]
            loss_list.append(loss)
        # LOGGER.debug("In compute_loss, loss list are: {}".format(loss_list))
        self.sync_loss_info(loss_list, suffix=current_suffix)

    def compute_forward_hess(self, data_instances, delta_s, host_forwards):
        """
        To compute Hessian matrix, y, s are needed.
        g = (1/N)*∑(wx - y) * x
        y = ∇2^F(w_t)s_t = g' * s = (1/N)*∑(x * s) * x
        define forward_hess = (1/N)*∑(x * s)
        """
        forwards = data_instances.mapValues(
            lambda v: (vec_dot(v.features, delta_s.coef_) + delta_s.intercept_))
        for host_forward in host_forwards:
            forwards = forwards.join(host_forward, lambda g, h: g + h)
        if self.use_sample_weight:
            forwards = forwards.join(data_instances, lambda h, d: h * d.weight)
        hess_vector = self.compute_gradient(data_instances,
                                            forwards,
                                            delta_s.fit_intercept)
        return forwards, np.array(hess_vector)


class Host(hetero_linear_model_gradient.Host, loss_sync.Host):

    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.host_forward,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.host_gradient,
                                     transfer_variables.host_optim_gradient)
        self._register_loss_sync(transfer_variables.host_loss_regular,
                                 transfer_variables.loss,
                                 transfer_variables.loss_intermediate)

    def compute_forwards(self, data_instances, model_weights):
        wx = data_instances.mapValues(
            lambda v: vec_dot(v.features, model_weights.coef_) + model_weights.intercept_)
        return wx

    def compute_half_g(self, data_instances, w, cipher, batch_index):
        half_g = data_instances.mapValues(
            lambda v: vec_dot(v.features, w.coef_) + w.intercept_)
        encrypt_half_g = cipher[batch_index].encrypt(half_g)
        return half_g, encrypt_half_g

    def compute_loss(self, model_weights, optimizer, n_iter_, batch_index, cipher_operator):
        '''
        Compute htero linr loss for:
            loss = (1/2N)*\\sum(wx-y)^2 where y is label, w is model weight and x is features

            Note: (wx - y)^2 = (wx_h)^2 + (wx_g - y)^2 + 2*(wx_h + (wx_g - y))
        '''

        current_suffix = (n_iter_, batch_index)
        self_wx_square = self.forwards.mapValues(lambda x: np.square(x)).reduce(reduce_add)
        en_wx_square = cipher_operator.encrypt(self_wx_square)
        self.remote_loss_intermediate(en_wx_square, suffix=current_suffix)

        loss_regular = optimizer.loss_norm(model_weights)
        if loss_regular is not None:
            en_loss_regular = cipher_operator.encrypt(loss_regular)
            self.remote_loss_regular(en_loss_regular, suffix=current_suffix)


class Arbiter(hetero_linear_model_gradient.Arbiter, loss_sync.Arbiter):
    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.guest_gradient,
                                     transfer_variables.host_gradient,
                                     transfer_variables.guest_optim_gradient,
                                     transfer_variables.host_optim_gradient)
        self._register_loss_sync(transfer_variables.loss)

    def compute_loss(self, cipher, n_iter_, batch_index):
        """
        Decrypt loss from guest
        """
        current_suffix = (n_iter_, batch_index)
        loss_list = self.sync_loss_info(suffix=current_suffix)
        de_loss_list = cipher.decrypt_list(loss_list)
        return de_loss_list
