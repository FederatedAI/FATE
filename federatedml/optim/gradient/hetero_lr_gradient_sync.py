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

from federatedml.framework.hetero.sync import gradient_sync
from federatedml.optim.gradient.logistic_gradient import HeteroLogisticGradientComputer
import numpy as np


class Guest(gradient_sync.Guest):
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

