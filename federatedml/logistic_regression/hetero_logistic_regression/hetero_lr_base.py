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
from federatedml.util import consts
from federatedml.util.transfer_variable.hetero_lr_transfer_variable import HeteroLRTransferVariable

class HeteroLRBase(BaseLogisticRegression):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroLogisticRegression'
        self.model_param_name = 'HeteroLogisticRegressionParam'
        self.model_meta_name = 'HeteroLogisticRegressionMeta'
        self.mode = consts.HETERO

    def _init_model(self, params):
        super(HeteroLRBase, self)._init_model(params)
        self.transfer_variable = HeteroLRTransferVariable()