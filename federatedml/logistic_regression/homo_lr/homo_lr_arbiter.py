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

import functools

from arch.api.utils import log_utils
from federatedml.evaluation import Evaluation
from federatedml.homo.procedure import aggregator, paillier_cipher
from federatedml.logistic_regression.homo_logsitic_regression.homo_lr_base import HomoLRBase
from federatedml.logistic_regression.logistic_regression_variables import LogisticRegressionVariables
from federatedml.model_selection import MiniBatch
from federatedml.optim.federated_aggregator.homo_federated_aggregator import HomoFederatedAggregator
from federatedml.optim.gradient import LogisticGradient, TaylorLogisticGradient
from federatedml.statistic import data_overview
from federatedml.util import consts
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


class HomoLRHost(HomoLRBase):
    def __init__(self):
        super(HomoLRHost, self).__init__()
        self.aggregator = HomoFederatedAggregator
        self.gradient_operator = None
        self.evaluator = Evaluation()
        self.re_encrypt_times = []  # Record the times needed for each host

        self.loss_history = []
        self.is_converged = False
        self.role = consts.GUEST
        self.aggregator = aggregator.Arbiter()
        self.lr_variables = None
        self.cipher = paillier_cipher.Arbiter()

    def _init_model(self, params):
        super()._init_model(params)
        self.cipher.register_paillier_cipher(self.transfer_variable)
        self.converge_flag_transfer = self.transfer_variable.converge_flag

    def fit(self, data_instances):
        host_ciphers = self.cipher.paillier_keygen(key_length=self.model_param.encrypt_param.key_length,
                                                   suffix=tuple('fit'))
        host_has_cipher_ids = list(host_ciphers.keys())
        self.re_encrypt_times = self.cipher.set_re_cipher_time(host_ciphers)
        max_iter = self.max_iter

        while self.n_iter_ < max_iter:
            self.cipher.re_cipher(iter_num=self.n_iter_,
                                  re_encrypt_times=self.re_encrypt_times,
                                  host_ciphers_dict=host_ciphers,
                                  re_encrypt_batches=self.re_encrypt_batches)
            self.aggregator.aggregate_and_broadcast(ciphers_dict=host_ciphers,
                                                    suffix=tuple(self.n_iter_))

            total_loss = self.aggregator.aggregate_loss(idx=host_has_cipher_ids,
                                                        suffix=tuple(self.n_iter_))
            # TODO: converge logic

            # self.converge_flag_transfer.remote(converge_flag, suffix=(iter_num,))

    def __init_model(self, data_instances):
        model_shape = data_overview.get_features_shape(data_instances)

        LOGGER.info("Initialized model shape is {}".format(model_shape))

        lr_variables = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        return lr_variables







