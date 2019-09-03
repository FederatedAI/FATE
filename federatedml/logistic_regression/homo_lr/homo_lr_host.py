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
from federatedml.framework.homo.procedure import aggregator
from federatedml.framework.homo.procedure import paillier_cipher
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
        self.loss_history = []
        self.is_converged = False
        self.role = consts.GUEST
        self.aggregator = aggregator.Host()
        self.lr_variables = None
        self.cipher = paillier_cipher.Host()

    def _init_model(self, params):
        super()._init_model(params)
        self.cipher.register_paillier_cipher(self.transfer_variable)
        if params.encrypt_params.method in [consts.PAILLIER]:
            self.use_encrypt = True
            self.gradient_operator = TaylorLogisticGradient()
            self.re_encrypt_batches = params.re_encrypt_batches
        else:
            self.use_encrypt = False
            self.gradient_operator = LogisticGradient()

    def fit(self, data_instances):

        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)

        pubkey = self.cipher.gen_paillier_pubkey(enable=self.use_encrypt, suffix=tuple('fit'))
        if self.use_encrypt:
            self.cipher_operator.set_public_key(pubkey)

        self.lr_variables = self.__init_model(data_instances)
        self.lr_variables = self.lr_variables.encrypted(cipher=self.cipher_operator)

        max_iter = self.max_iter
        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)
        iter_loss = 0
        batch_num = 0
        while self.n_iter_ < max_iter:
            batch_data_generator = mini_batch_obj.mini_batch_data_generator()

            for batch_data in batch_data_generator:
                n = batch_data.count()
                f = functools.partial(self.gradient_operator.compute,
                                      coef=self.lr_variables.coef_,
                                      intercept=self.lr_variables.intercept_,
                                      fit_intercept=self.fit_intercept)
                grad_loss = batch_data.mapPartitions(f)
                grad, loss = grad_loss.reduce(fate_operator.reduce_add)
                grad /= n
                self.lr_variables = self.optimizer.update_model(self.lr_variables, grad)
                if not self.use_encrypt:
                    loss /= n
                    iter_loss += (loss + self.optimizer.loss_norm(self.lr_variables))
                batch_num += 1
                if self.use_encrypt and self.n_iter_ % self.re_encrypt_batches == 0:
                    w = self.cipher.re_cipher(w=self.lr_variables.for_remote(),
                                              iter_num=self.n_iter_,
                                              batch_iter_num=batch_num)
                    self.lr_variables = LogisticRegressionVariables(w, self.fit_intercept)

            self.aggregator.send_model_for_aggregate(self.lr_variables.for_remote(), self.n_iter_)
            if self.use_encrypt:
                iter_loss /= batch_num
                self.callback_loss(self.n_iter_, iter_loss)
                self.loss_history.append(iter_loss)
                self.aggregator.send_loss(iter_loss, self.n_iter_)
            weight = self.aggregator.get_aggregated_model(self.n_iter_)
            self.lr_variables = LogisticRegressionVariables(weight, self.fit_intercept)

    def __init_model(self, data_instances):
        model_shape = data_overview.get_features_shape(data_instances)

        LOGGER.info("Initialized model shape is {}".format(model_shape))

        lr_variables = self.initializer.init_model(model_shape, init_params=self.init_param_obj)

        return lr_variables
