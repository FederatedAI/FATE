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


from arch.api.utils import log_utils
from federatedml.framework.hetero_lr_utils.sync import three_parties_sync
from federatedml.logistic_regression.logistic_regression_variables import LogisticRegressionVariables

LOGGER = log_utils.getLogger()


class Guest(three_parties_sync.Guest):
    def __init__(self):
        self.batch_index = 0
        self.n_iter_ = 0

    def _register_intermediate_transfer(self, transfer_variables):
        self.host_loss_regular_transfer = transfer_variables.host_loss_regular
        self.loss_transfer = transfer_variables.loss

    def _register_attrs(self, lr_model):
        if lr_model.model_param.penalty in ['l1', 'l2']:
            self.has_penalty = True

    def register_loss_procedure(self, transfer_variables, lr_model):
        self._register_intermediate_transfer(transfer_variables)
        self._register_attrs(lr_model)

    def renew_current_info(self, n_iter, batch_index):
        self.n_iter_ = n_iter
        self.batch_index = batch_index

    def apply_procedure(self, loss):
        current_suffix = (self.n_iter_, self.batch_index)

        if self.has_penalty:
            en_host_loss_regular = self.host_to_guest(transfer_variables=(self.host_loss_regular_transfer,),
                                                      suffix=current_suffix)[0]
            LOGGER.info("Get host_loss_regular from Host")
            loss += en_host_loss_regular

        self.guest_to_arbiter(variables=(loss,),
                              transfer_variables=(self.loss_transfer,),
                              suffix=current_suffix)


