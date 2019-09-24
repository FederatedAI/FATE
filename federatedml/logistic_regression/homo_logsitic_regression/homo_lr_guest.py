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
from federatedml.framework.homo.procedure import aggregator, predict_procedure
from federatedml.logistic_regression.homo_logsitic_regression.homo_lr_base import HomoLRBase
from federatedml.logistic_regression.logistic_regression_weights import LogisticRegressionWeights
from federatedml.model_selection import MiniBatch
from federatedml.optim.gradient.logistic_gradient import LogisticGradient
from federatedml.util import consts
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


class HomoLRGuest(HomoLRBase):
    def __init__(self):
        super(HomoLRGuest, self).__init__()
        self.gradient_operator = LogisticGradient()
        self.loss_history = []
        self.role = consts.GUEST
        self.aggregator = aggregator.Guest()
        self.predict_procedure = predict_procedure.Guest()

    def _init_model(self, params):
        super()._init_model(params)
        self.predict_procedure.register_predict_sync(self.transfer_variable, self)

    def fit(self, data_instances):

        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)

        self.lr_weights = self._init_model_variables(data_instances)

        max_iter = self.max_iter
        total_data_num = data_instances.count()
        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)

        lr_weights = self.lr_weights
        while self.n_iter_ < max_iter:
            batch_data_generator = mini_batch_obj.mini_batch_data_generator()

            self.optimizer.set_iters(self.n_iter_)
            if self.n_iter_ > 0 and self.n_iter_ % self.aggregate_iters == 0:
                weight = self.aggregator.aggregate_then_get(lr_weights, degree=total_data_num,
                                                            suffix=self.n_iter_)
                self.lr_weights = LogisticRegressionWeights(weight.unboxed, self.fit_intercept)
                loss = self._compute_loss(data_instances)
                self.aggregator.send_loss(loss, degree=total_data_num, suffix=(self.n_iter_,))
                self.is_converged = self.aggregator.get_converge_status(suffix=(self.n_iter_,))
                LOGGER.info("n_iters: {}, loss: {} converge flag is :{}".format(self.n_iter_, loss, self.is_converged))
                if self.is_converged:
                    break
                lr_weights = self.lr_weights

            batch_num = 0
            for batch_data in batch_data_generator:
                n = batch_data.count()
                LOGGER.debug("In each batch, lr_weight: {}".format(lr_weights.unboxed))
                f = functools.partial(self.gradient_operator.compute_gradient,
                                      coef=lr_weights.coef_,
                                      intercept=lr_weights.intercept_,
                                      fit_intercept=self.fit_intercept)
                grad = batch_data.mapPartitions(f).reduce(fate_operator.reduce_add)
                LOGGER.debug('iter: {}, batch_index: {}, grad: {}, n: {}'.format(
                    self.n_iter_, batch_num, grad, n))
                grad /= n
                lr_weights = self.optimizer.update_model(lr_weights, grad, has_applied=False)
                batch_num += 1
            self.n_iter_ += 1

    def predict(self, data_instances):
        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)
        predict_result = self.predict_procedure.start_predict(data_instances,
                                                              self.lr_weights,
                                                              self.model_param.predict_param.threshold)
        return predict_result


