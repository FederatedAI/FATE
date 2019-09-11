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
from federatedml.framework.hetero.sync import three_parties_sync
from federatedml.framework.weights import ListWeights
from federatedml.optim.gradient.logistic_gradient import HeteroLogisticGradientComputer

# from federatedml.statistic.data_overview import rubbish_clear
# from federatedml.statistic import data_overview

LOGGER = log_utils.getLogger()


class Base(HeteroLogisticGradientComputer):
    def __init__(self):
        self.n_iter_ = 0
        self.batch_index = 0
        self.cipher_operator = None

        # func part
        self.compute_wx = None
        self.update_local_model = None


class Guest(three_parties_sync.Guest, Base):
    def __init__(self):
        super().__init__()
        self.encrypted_calculator = None
        self.guest_forward = None

    def _register_intermediate_transfer(self, transfer_variables):
        self.host_forward_dict_transfer = transfer_variables.host_forward_dict
        self.fore_gradient_transfer = transfer_variables.fore_gradient
        self.guest_gradient_transfer = transfer_variables.guest_gradient
        self.guest_optim_gradient_transfer = transfer_variables.guest_optim_gradient

    def register_func(self, lr_model):
        self.compute_wx = lr_model.compute_wx
        self.update_local_model = lr_model.update_local_model

    def register_attrs(self, lr_model):
        self.encrypted_calculator = lr_model.encrypted_calculator
        self.cipher_operator = lr_model.cipher_operator

    def register_gradient_procedure(self, transfer_variables):
        self._register_intermediate_transfer(transfer_variables)

    def renew_current_info(self, n_iter, batch_index):
        self.n_iter_ = n_iter
        self.batch_index = batch_index

    def computer_intermediate(self, data_instances, lr_weights):
        """
        Compute W * X + b and (W * X + b)^2, where X is the input data, W is the coefficient of lr,
        and b is the interception
        Parameters
        ----------
        data_instances: DTable of Instance, input data

        lr_weights: LogisticRegressionVariables
            Stores coef_ and intercept_ of lr

        """
        wx = self.compute_wx(data_instances, lr_weights.coef_, lr_weights.intercept_)

        en_wx = self.encrypted_calculator[self.batch_index].encrypt(wx)
        wx_square = wx.mapValues(lambda v: np.square(v))
        en_wx_square = self.encrypted_calculator[self.batch_index].encrypt(wx_square)

        en_wx_join_en_wx_square = en_wx.join(en_wx_square, lambda wx, wx_square: (wx, wx_square))
        self.guest_forward = en_wx_join_en_wx_square.join(wx, lambda e, wx: (e[0], e[1], wx))

        # temporary resource recovery and will be removed in the future
        # rubbish_list = [en_wx, wx_square, en_wx_square, en_wx_join_en_wx_square]
        # rubbish_clear(rubbish_list)

    def aggregate_forward(self, host_forward):
        """
        Compute (en_wx_g + en_wx_h)^2 = en_wx_g^2 + en_wx_h^2 + 2 * wx_g * en_wx_h ,
         where en_wx_g is the encrypted W * X + b of guest, wx_g is unencrypted W * X + b,
        and en_wx_h is the encrypted W * X + b of host.
        Parameters
        ----------
        host_forward: DTable, include encrypted W * X and (W * X)^2

        Returns
        ----------
        aggregate_forward_res
        list
            include W * X and (W * X)^2 federate with guest and host
        """
        aggregate_forward_res = self.guest_forward.join(host_forward,
                                                        lambda g, h: (g[0] + h[0], g[1] + h[1] + 2 * g[2] * h[0]))

        en_aggregate_wx = aggregate_forward_res.mapValues(lambda v: v[0])
        en_aggregate_wx_square = aggregate_forward_res.mapValues(lambda v: v[1])

        # self.rubbish_bin.append(aggregate_forward_res)
        return en_aggregate_wx, en_aggregate_wx_square

    def apply_procedure(self, data_instances, lr_weights):
        current_suffix = (self.n_iter_, self.batch_index)

        self.computer_intermediate(data_instances, lr_weights)
        host_forward = self.host_to_guest((self.host_forward_dict_transfer,),
                                          suffix=current_suffix)[0]
        LOGGER.info("Get host_forward from host")
        en_aggregate_wx, en_aggregate_wx_square = self.aggregate_forward(host_forward=host_forward)
        fore_gradient = self.compute_fore_gradient(data_instances, en_aggregate_wx)
        self.guest_to_host(variables=(fore_gradient,),
                           transfer_variables=(self.fore_gradient_transfer,),
                           suffix=current_suffix)
        LOGGER.info("Remote fore_gradient to Host")
        guest_gradient, loss = self.compute_gradient_and_loss(data_instances,
                                                              fore_gradient,
                                                              en_aggregate_wx,
                                                              en_aggregate_wx_square,
                                                              lr_weights.fit_intercept)
        gradient_var = ListWeights(guest_gradient)

        self.guest_to_arbiter(variables=(gradient_var.for_remote(),),
                              transfer_variables=(self.guest_gradient_transfer,),
                              suffix=current_suffix)
        LOGGER.info("Remote guest_gradient to arbiter")

        optim_guest_gradient = self.arbiter_to_guest(transfer_variables=(self.guest_optim_gradient_transfer,),
                                                     suffix=current_suffix)[0]
        LOGGER.info("Get optim_guest_gradient from arbiter")

        training_info = {"iteration": self.n_iter_, "batch_index": self.batch_index}
        self.update_local_model(fore_gradient, data_instances, lr_weights.coef_, **training_info)

        # self.rubbish_bin.extend([en_aggregate_wx,
        #                          host_forward,
        #                          en_aggregate_wx_square,
        #                          fore_gradient,
        #                          self.guest_forward
        #                          ])
        # data_overview.rubbish_clear(self.rubbish_bin)
        # self.rubbish_bin = []

        return optim_guest_gradient, loss


class Host(three_parties_sync.Host, Base):
    def __init__(self):
        super().__init__()
        self.encrypted_calculator = None

    def _register_intermediate_transfer(self, transfer_variables):
        self.host_forward_dict_transfer = transfer_variables.host_forward_dict
        self.fore_gradient_transfer = transfer_variables.fore_gradient
        self.host_gradient_transfer = transfer_variables.host_gradient
        self.host_optim_gradient_transfer = transfer_variables.host_optim_gradient

    def register_func(self, lr_model):
        self.compute_wx = lr_model.compute_wx
        self.update_local_model = lr_model.update_local_model

    def register_attrs(self, lr_model):
        self.encrypted_calculator = lr_model.encrypted_calculator
        self.cipher_operator = lr_model.cipher_operator

    def register_gradient_procedure(self, transfer_variables):
        self._register_intermediate_transfer(transfer_variables)

    def renew_current_info(self, n_iter, batch_index):
        self.n_iter_ = n_iter
        self.batch_index = batch_index

    def computer_intermediate(self, data_instances, lr_weights):
        """
        Compute W * X + b and (W * X + b)^2, where X is the input data, W is the coefficient of lr,
        and b is the interception
        Parameters
        ----------
        data_instances: DTable of Instance, input data

        lr_weights: LogisticRegressionVariables
            Stores coef_ and intercept_ of lr

        """
        wx = self.compute_wx(data_instances, lr_weights.coef_, lr_weights.intercept_)

        en_wx = self.encrypted_calculator[self.batch_index].encrypt(wx)
        wx_square = wx.mapValues(lambda v: np.square(v))
        en_wx_square = self.encrypted_calculator[self.batch_index].encrypt(wx_square)
        host_forward = en_wx.join(en_wx_square, lambda wx, wx_square: (wx, wx_square))

        return host_forward

    def apply_procedure(self, data_instances, lr_weights):
        current_suffix = (self.n_iter_, self.batch_index)

        host_forward = self.computer_intermediate(data_instances, lr_weights)
        self.host_to_guest(variables=(host_forward,),
                           transfer_variables=(self.host_forward_dict_transfer,),
                           suffix=current_suffix)
        LOGGER.info("Remote host_forward to guest")

        fore_gradient = self.guest_to_host(transfer_variables=(self.fore_gradient_transfer,),
                                           suffix=current_suffix)[0]

        LOGGER.info("Get fore_gradient from guest")

        host_gradient = self.compute_gradient(data_instances,
                                              fore_gradient,
                                              fit_intercept=False)

        gradient_var = ListWeights(host_gradient)

        self.host_to_arbiter(variables=(gradient_var.for_remote(),),
                             transfer_variables=(self.host_gradient_transfer,),
                             suffix=current_suffix)
        LOGGER.info("Remote host_gradient to arbiter")

        optim_host_gradient = self.arbiter_to_host(transfer_variables=(self.host_optim_gradient_transfer,),
                                                   suffix=current_suffix)[0]
        LOGGER.info("Get optim_guest_gradient from arbiter")

        training_info = {"iteration": self.n_iter_, "batch_index": self.batch_index}
        self.update_local_model(fore_gradient, data_instances, lr_weights.coef_, **training_info)

        return optim_host_gradient


class Arbiter(three_parties_sync.Arbiter, Base):
    def __init__(self):
        super().__init__()
        self.optimizer = None

    def _register_intermediate_transfer(self, transfer_variables):
        self.guest_gradient_transfer = transfer_variables.guest_gradient
        self.host_gradient_transfer = transfer_variables.host_gradient
        self.guest_optim_gradient_transfer = transfer_variables.guest_optim_gradient
        self.host_optim_gradient_transfer = transfer_variables.host_optim_gradient

    def register_attrs(self, lr_model):
        self.optimizer = lr_model.optimizer
        self.cipher_operator = lr_model.cipher_operator

    def register_gradient_procedure(self, transfer_variables):
        self._register_intermediate_transfer(transfer_variables)

    def renew_current_info(self, n_iter, batch_index):
        self.n_iter_ = n_iter
        self.batch_index = batch_index

    @staticmethod
    def separate(value, size_list):
        """
        Separate value in order to several set according size_list
        Parameters
        ----------
        value: list or ndarray, input data
        size_list: list, each set size

        Returns
        ----------
        list
            set after separate
        """
        separate_res = []
        cur = 0
        for size in size_list:
            separate_res.append(value[cur:cur + size])
            cur += size
        return separate_res

    def apply_procedure(self):
        current_suffix = (self.n_iter_, self.batch_index)

        host_gradient = self.host_to_arbiter(transfer_variables=(self.host_gradient_transfer,),
                                             suffix=current_suffix)[0].parameter
        LOGGER.info("Get host_gradient from Host")

        guest_gradient = self.guest_to_arbiter(transfer_variables=(self.guest_gradient_transfer,),
                                               suffix=current_suffix)[0].parameter
        LOGGER.info("Get guest_gradient from Guest")

        host_gradient, guest_gradient = np.array(host_gradient), np.array(guest_gradient)
        gradient = np.hstack((host_gradient, guest_gradient))

        gradient_var = ListWeights(gradient)
        gradient_var.decrypted(self.cipher_operator, True)
        grad = gradient_var.for_remote().parameters
        delta_grad = self.optimizer.apply_gradients(grad)
        separate_optim_gradient = self.separate(delta_grad,
                                                [host_gradient.shape[0],
                                                 guest_gradient.shape[0]])
        host_optim_gradient = separate_optim_gradient[0]
        guest_optim_gradient = separate_optim_gradient[1]

        self.arbiter_to_host(variables=(host_optim_gradient,),
                             transfer_variables=(self.host_optim_gradient_transfer,),
                             suffix=current_suffix)

        self.arbiter_to_guest(variables=(guest_optim_gradient,),
                              transfer_variables=(self.guest_optim_gradient_transfer,),
                              suffix=current_suffix)
