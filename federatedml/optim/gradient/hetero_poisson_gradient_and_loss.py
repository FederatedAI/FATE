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
        self._register_gradient_sync(transfer_variables.host_forward,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.guest_gradient,
                                     transfer_variables.guest_optim_gradient)
        self._register_loss_sync(transfer_variables.host_loss_regular,
                                 transfer_variables.loss,
                                 transfer_variables.loss_intermediate)

    def compute_and_aggregate_forwards(self, data_instances, model_weights, encrypted_calculator,
                                       batch_index, offset=None):
        '''
        Compute gradients:
        gradient = (1/N) * \sum(exp(wx) - y) * x

        Define exp(wx) as mu, named it as guest_forward or host_forward
        Define (mu-y) as fore_gradient
        Then, gradient = fore_gradeint * x

        '''
        if offset is None:
            raise ValueError("Offset should be provided when compute poisson forwards")
        mu = data_instances.join(offset, lambda d, m: np.exp(vec_dot(d.features, model_weights.coef_)
                                                             + model_weights.intercept_ + m))
        self.forwards = mu

        self.aggregated_forwards = self.forwards.join(self.host_forwards[0], lambda g, h: g * h)
        fore_gradient = self.aggregated_forwards.join(data_instances, lambda mu, d: mu - d.label)
        return fore_gradient

    def compute_loss(self, data_instances, model_weights, n_iter_, batch_index, offset, loss_norm=None):
        '''
        Compute hetero poisson loss:
            loss = sum(exp(mu_g)*exp(mu_h) - y(wx_g + wx_h) + log(exposure))

        Parameters:
        ___________
        data_instances: DTable, input data

        model_weights: model weight object, stores intercept_ and coef_

        n_iter_: int, current number of iter.

        batch_index: int, use to obtain current encrypted_calculator index

        offset: log(exposure)

        loss_norm: penalty term, default to None

        '''
        current_suffix = (n_iter_, batch_index)
        n = data_instances.count()
        guest_wx_y = data_instances.join(offset,
            lambda v, m: (vec_dot(v.features, model_weights.coef_) + model_weights.intercept_ + m, v.label))
        loss_list = []
        host_wxs = self.get_host_loss_intermediate(current_suffix)
        if loss_norm is not None:
            host_loss_regular = self.get_host_loss_regular(suffix=current_suffix)
        else:
            host_loss_regular = []

        if len(self.host_forwards) > 1:
            raise ValueError("More than one host exists. Poisson regression does not support multi-host.")

        host_mu = self.host_forwards[0]
        host_wx = host_wxs[0]
        loss_wx = guest_wx_y.join(host_wx, lambda g, h: g[1] * (g[0] + h)).reduce(reduce_add)
        loss_mu = self.forwards.join(host_mu, lambda g, h: g * h).reduce(reduce_add)
        loss = (loss_mu - loss_wx) / n
        if loss_norm is not None:
            loss = loss + loss_norm + host_loss_regular[0]
        loss_list.append(loss)
        self.sync_loss_info(loss_list, suffix=current_suffix)


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
        mu = data_instances.mapValues(
            lambda v: np.exp(vec_dot(v.features, model_weights.coef_) + model_weights.intercept_))
        return mu

    def compute_loss(self, data_instances, model_weights, encrypted_calculator,
                     optimizer, n_iter_, batch_index, cipher_operator):
        '''
        Compute hetero poisson loss:
            h_loss = sum(exp(mu_h))

        Parameters:
        ___________
        data_instances: DTable, input data

        model_weights: model weight object, stores intercept_ and coef_

        encrypted_calculator: ecnrypted calculator object

        optimizer: optimizer object

        n_iter_: int, current number of iter.

        batch_index: int, use to obtain current encrypted_calculator index

        cipher_operator: cipher for encrypt intermediate loss and loss_regular

        '''
        current_suffix = (n_iter_, batch_index)
        self_wx = data_instances.mapValues(lambda v: vec_dot(v.features, model_weights.coef_) + model_weights.intercept_)
        en_wx = encrypted_calculator[batch_index].encrypt(self_wx)
        self.remote_loss_intermediate(en_wx, suffix=current_suffix)

        loss_regular = optimizer.loss_norm(model_weights)
        if loss_regular is None:
            en_loss_regular = loss_regular
        else:
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
        '''
        Decrypt loss from guest
        '''
        current_suffix = (n_iter_, batch_index)
        loss_list = self.sync_loss_info(suffix=current_suffix)
        de_loss_list = cipher.decrypt_list(loss_list)
        return de_loss_list
