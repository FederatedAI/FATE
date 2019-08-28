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

import numpy as np

from arch.api import federation
from arch.api.utils import log_utils
from federatedml.evaluation import Evaluation
from federatedml.logistic_regression.homo_logsitic_regression.homo_lr_base import HomoLRBase
from federatedml.model_selection import MiniBatch
from federatedml.optim import Initializer
from federatedml.optim import activation
from federatedml.optim.gradient import LogisticGradient
from fate_flow.entity.metric import MetricMeta
from federatedml.optim.federated_aggregator.homo_federated_aggregator import HomoFederatedAggregator
from fate_flow.entity.metric import Metric
from federatedml.homo.procedure import aggregator
from federatedml.util import consts
from federatedml.statistic import data_overview
from federatedml.logistic_regression.logistic_regression_variables import LogisticRegressionVariables as LRParam
LOGGER = log_utils.getLogger()


class HomoLRGuest(HomoLRBase, aggregator.Guest):
    def __init__(self):
        super(HomoLRGuest, self).__init__()
        self.aggregator = HomoFederatedAggregator
        self.gradient_operator = LogisticGradient()

        self.classes_ = [0, 1]

        self.evaluator = Evaluation()
        self.loss_history = []
        self.is_converged = False
        self.role = consts.GUEST
        self.lr_param = None

    def _init_model(self, params):
        super()._init_model(params)
        self.register_aggregator(self.transfer_variable)
        self.initialize_aggregator(params.party_weight)

    def fit(self, data_instances):

        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)

        self.lr_param = self.__init_model(data_instances)

        max_iter = self.max_iter
        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)
        iter_loss = 0
        batch_num = 0
        while self.n_iter_ < max_iter:
            batch_data_generator = mini_batch_obj.mini_batch_data_generator()

            for batch_data in batch_data_generator:
                n = batch_data.count()
                f = functools.partial(self.gradient_operator.compute,
                                      coef=self.lr_param.coef_,
                                      intercept=self.lr_param.intercept_,
                                      fit_intercept=self.fit_intercept)
                grad_loss = batch_data.mapPartitions(f)
                grad, loss = grad_loss.reduce(self.aggregator.aggregate_grad_loss)

                grad /= n
                loss /= n
                self.lr_param = self.optimizer.apply_gradients(self.lr_param, grad)
                iter_loss += (loss + self.optimizer.loss_norm(self.lr_param))
                batch_num += 1
            iter_loss /= batch_num
            self.callback_loss(self.n_iter_, iter_loss)
            self.loss_history.append(iter_loss)
            self.send_model(self.lr_param.for_remote(), self.n_iter_)
            self.send_loss(iter_loss, self.n_iter_)


    def __init_model(self, data_instances):
        model_shape = data_overview.get_features_shape(data_instances)

        LOGGER.info("Initialized model shape is {}".format(model_shape))

        lr_param = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        return lr_param
