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
from federatedml.optim.gradient import hetero_lr_gradient_sync
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class Guest(hetero_lr_gradient_sync.Guest):
    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.host_forward_dict,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.guest_gradient,
                                     transfer_variables.guest_optim_gradient)

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
        # wx_square = wx.mapValues(lambda v: np.square(v))
        # en_wx_square = encrypted_calculator[batch_index].encrypt(wx_square)

        # en_wx_join_en_wx_square = en_wx.join(en_wx_square, lambda wx, wx_square: (wx, wx_square))
        # guest_forward = en_wx_join_en_wx_square.join(wx, lambda e, wx: (e[0], e[1], wx))
        return en_wx

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
        # aggregate_forward_res = guest_forward.join(host_forward,
        #                                            lambda g, h: (g[0] + h[0], g[1] + h[1] + 2 * g[2] * h[0]))

        aggregate_forward_res = guest_forward.join(host_forward,
                                                   lambda g, h: g + h)

        # en_aggregate_wx = aggregate_forward_res.mapValues(lambda v: v[0])
        # en_aggregate_wx_square = aggregate_forward_res.mapValues(lambda v: v[1])

        # self.rubbish_bin.append(aggregate_forward_res)
        # return en_aggregate_wx, en_aggregate_wx_square
        return aggregate_forward_res

    def compute_lr_gradient(self, data_instances, lr_variables,
                            encrypted_calculator,
                            n_iter_, batch_index):
        """
        Compute gradients.
        gradient = (1/N)*∑(1/2*ywx-1)*1/2yx = (1/N)*∑(0.25 * wx - 0.5 * y) * x, where y = 1 or -1

        Define ∑wx as guest_forward or host_forward
        Define ∑(0.25 * wx - 0.5 * y) as fore_gradient

        Then, gradient = fore_gradient * x


        Parameters
        ----------
        data_instances: DTable of Instance, input data

        lr_variables: LogisticRegressionVariables
            Stores coef_ and intercept_ of lr

        encrypted_calculator: Use for different encrypted methods

        batch_index: int, use to obtain current encrypted_calculator

        """
        current_suffix = (n_iter_, batch_index)
        wx = data_instances.mapValues(lambda v: np.dot(v.features, lr_variables.coef_) + lr_variables.intercept_)
        aggregate_forward_res = encrypted_calculator[batch_index].encrypt(wx)
        host_forwards = self.get_host_forward(suffix=current_suffix)
        for host_forward in host_forwards:
            aggregate_forward_res = aggregate_forward_res.join(host_forward, lambda g, h: g + h)
        fore_gradient = aggregate_forward_res.join(data_instances, lambda wx, d: 0.25 * wx - 0.5 * d.label)

        self.remote_fore_gradient(fore_gradient, suffix=current_suffix)

        unilateral_gradient = self.compute_gradient(data_instances,
                                                    fore_gradient,
                                                    lr_variables.fit_intercept)
        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        return optimized_gradient


class Host(hetero_lr_gradient_sync.Host):
    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.host_forward_dict,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.host_gradient,
                                     transfer_variables.host_optim_gradient)

    def compute_lr_gradient(self, data_instances, lr_variables,
                            encrypted_calculator,
                            n_iter_, batch_index):
        """
        Compute gradients.
        gradient = (1/N)*∑(1/2*ywx-1)*1/2yx = (1/N)*∑(0.25 * wx - 0.5 * y) * x, where y = 1 or -1

        Define ∑(0.25 * wx - 0.5 * y) as fore_gradient

        Parameters
        ----------
        data_instances: DTable of Instance, input data

        lr_variables: LogisticRegressionVariables
            Stores coef_ and intercept_ of lr

        encrypted_calculator: Use for different encrypted methods

        n_iter_: int, current iter nums

        batch_index: int, use to obtain current encrypted_calculator

        """
        current_suffix = (n_iter_, batch_index)
        wx = data_instances.mapValues(lambda v: np.dot(v.features, lr_variables.coef_) + lr_variables.intercept_)
        host_forward = encrypted_calculator[batch_index].encrypt(wx)
        self.remote_host_forward(host_forward, suffix=current_suffix)

        fore_gradient = self.get_fore_gradient(suffix=current_suffix)

        unilateral_gradient = self.compute_gradient(data_instances,
                                                    fore_gradient,
                                                    lr_variables.fit_intercept)
        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        return optimized_gradient


class Arbiter(hetero_lr_gradient_sync.Arbiter):
    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.guest_gradient,
                                     transfer_variables.host_gradient,
                                     transfer_variables.guest_optim_gradient,
                                     transfer_variables.host_optim_gradient)

    def compute_lr_gradient(self, data_instances, lr_variables,
                            encrypted_calculator,
                            n_iter_, batch_index):
        """
        Compute gradients.
        gradient = (1/N)*∑(1/2*ywx-1)*1/2yx = (1/N)*∑(0.25 * wx - 0.5 * y) * x, where y = 1 or -1

        Received

        Parameters
        ----------
        data_instances: DTable of Instance, input data

        lr_variables: LogisticRegressionVariables
            Stores coef_ and intercept_ of lr

        encrypted_calculator: Use for different encrypted methods

        n_iter_: int, current iter nums

        batch_index: int, use to obtain current encrypted_calculator

        """
        current_suffix = (n_iter_, batch_index)
        wx = data_instances.mapValues(lambda v: np.dot(v.features, lr_variables.coef_) + lr_variables.intercept_)
        host_forward = encrypted_calculator[batch_index].encrypt(wx)
        self.remote_host_forward(host_forward, suffix=current_suffix)

        fore_gradient = self.get_fore_gradient(suffix=current_suffix)

        unilateral_gradient = self.compute_gradient(data_instances,
                                                    fore_gradient,
                                                    lr_variables.fit_intercept)
        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        return optimized_gradient
