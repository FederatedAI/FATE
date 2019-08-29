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

from federatedml.logistic_regression.base_logistic_regression import BaseLogisticRegression
from federatedml.util.transfer_variable.homo_lr_transfer_variable import HomoLRTransferVariable
from federatedml.optim import Optimizer
from federatedml.util import consts
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class HomoLRBase(BaseLogisticRegression):
    def __init__(self):
        super(HomoLRBase, self).__init__()
        self.model_name = 'HomoLogisticRegression'
        self.model_param_name = 'HomoLogisticRegressionParam'
        self.model_meta_name = 'HomoLogisticRegressionMeta'
        self.mode = consts.HOMO
        self.aggregator = None

    def _init_model(self, params):
        super(HomoLRBase, self)._init_model(params)
        self.transfer_variable = HomoLRTransferVariable()
        self.aggregator.register_aggregator(self.transfer_variable)
        self.aggregator.initialize_aggregator(params.party_weight)

    @property
    def use_loss(self):
        if self.model_param.converge_func == 'weight_diff':
            return False
        return True

