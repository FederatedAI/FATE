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
from federatedml.linear_model.linear_regression.base_linear_regression import BaseLinearRegression
from federatedml.optim.gradient.hetero_sqn_gradient import sqn_factory
from federatedml.transfer_variable.transfer_class.hetero_linr_transfer_variable import HeteroLinRTransferVariable
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroLinRBase(BaseLinearRegression):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroLinearRegression'
        self.model_param_name = 'HeteroLinearRegressionParam'
        self.model_meta_name = 'HeteroLinearRegressionMeta'
        self.mode = consts.HETERO
        self.aggregator = None
        self.cipher = None
        self.batch_generator = None
        self.gradient_loss_operator = None
        self.converge_procedure = None

    def _init_model(self, params):
        super(HeteroLinRBase, self)._init_model(params)
        self.transfer_variable = HeteroLinRTransferVariable()
        self.cipher.register_paillier_cipher(self.transfer_variable)
        self.converge_procedure.register_convergence(self.transfer_variable)
        self.batch_generator.register_batch_generator(self.transfer_variable)
        self.gradient_loss_operator.register_gradient_procedure(self.transfer_variable)

        if params.optimizer == 'sqn':
            gradient_loss_operator = sqn_factory(self.role, params.sqn_param)
            gradient_loss_operator.register_gradient_computer(self.gradient_loss_operator)
            gradient_loss_operator.register_transfer_variable(self.transfer_variable)
            self.gradient_loss_operator = gradient_loss_operator
            LOGGER.debug("In _init_model, optimizer: {}, gradient_loss_operator: {}".format(
                params.optimizer, self.gradient_loss_operator
            ))

    def transform(self, data_inst):
        return data_inst
