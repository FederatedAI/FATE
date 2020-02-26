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
from federatedml.framework.homo.procedure import aggregator
from federatedml.model_base import ModelBase
from federatedrec.param.matrix_factorization_param import HeteroMatrixParam
from federatedrec.transfer_variable.transfer_class.hetero_mf_transfer_variable import \
    HeteroMFTransferVariable

LOGGER = log_utils.getLogger()


class HeteroMFBase(ModelBase):
    """
    Hetero Matrix Factorization Base Class.
    """
    def __init__(self):
        super(HeteroMFBase, self).__init__()
        self.model_param_name = 'MatrixModelParam'
        self.model_meta_name = 'MatrixModelMeta'

        self.model_param = HeteroMatrixParam()
        self.aggregator = None
        self.user_ids_sync = None
        self.params = None
        self.transfer_variable = None
        self.aggregator_iter = None
        self.max_iter = None


    def _iter_suffix(self):
        return self.aggregator_iter,

    def _init_model(self, params):
        super(HeteroMFBase, self)._init_model(params)
        self.params = params
        self.transfer_variable = HeteroMFTransferVariable()
        secure_aggregate = params.secure_aggregate
        self.aggregator = aggregator.with_role(role=self.role,
                                               transfer_variable=self.transfer_variable,
                                               enable_secure_aggregate=secure_aggregate)
        self.max_iter = params.max_iter
        self.aggregator_iter = 0

    @staticmethod
    def extract_ids(data_instances):
        """
        Extract user ids and item ids from data instances.
        :param data_instances: Dtable of input data.
        :return: extracted user ids and item ids.
        """
        user_ids = data_instances.map(lambda k, v: (v.features.get_data(0), None))
        item_ids = data_instances.map(lambda k, v: (v.features.get_data(1), None))
        return user_ids, item_ids
