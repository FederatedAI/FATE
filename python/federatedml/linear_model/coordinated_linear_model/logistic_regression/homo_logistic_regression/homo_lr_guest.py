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

import copy
import functools

from federatedml.framework.homo.procedure import aggregator
from federatedml.linear_model.linear_model_weight import LinearModelWeights as LogisticRegressionWeights
from federatedml.linear_model.coordinated_linear_model.logistic_regression.\
    homo_logistic_regression.homo_lr_base import HomoLRBase
from federatedml.model_selection import MiniBatch
from federatedml.optim import activation
from federatedml.optim.gradient.homo_lr_gradient import LogisticGradient
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.util.fate_operator import vec_dot
from federatedml.util.io_check import assert_io_num_rows_equal


class HomoLRGuest(HomoLRBase):
    def __init__(self):
        super(HomoLRGuest, self).__init__()
        self.gradient_operator = LogisticGradient()
        self.loss_history = []
        self.role = consts.GUEST
        # self.aggregator = aggregator.Guest()

    def _init_model(self, params):
        super()._init_model(params)

    def fit_binary(self, data_instances, validate_data=None):
        self.aggregator = aggregator.Guest()
        self.aggregator.register_aggregator(self.transfer_variable)

        self.callback_list.on_train_begin(data_instances, validate_data)

        # validation_strategy = self.init_validation_strategy(data_instances, validate_data)
        if not self.component_properties.is_warm_start:
            self.model_weights = self._init_model_variables(data_instances)
        else:
            self.callback_warm_start_init_iter(self.n_iter_)

        max_iter = self.max_iter
        # total_data_num = data_instances.count()
        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)
        model_weights = self.model_weights

        degree = 0
        self.prev_round_weights = copy.deepcopy(model_weights)

        while self.n_iter_ < max_iter + 1:
            self.callback_list.on_epoch_begin(self.n_iter_)
            batch_data_generator = mini_batch_obj.mini_batch_data_generator()

            self.optimizer.set_iters(self.n_iter_)
            if ((self.n_iter_ + 1) % self.aggregate_iters == 0) or self.n_iter_ == max_iter:
                weight = self.aggregator.aggregate_then_get(model_weights, degree=degree,
                                                            suffix=self.n_iter_)

                self.model_weights = LogisticRegressionWeights(weight.unboxed, self.fit_intercept)

                # store prev_round_weights after aggregation
                self.prev_round_weights = copy.deepcopy(self.model_weights)
                # send loss to arbiter
                loss = self._compute_loss(data_instances, self.prev_round_weights)
                self.aggregator.send_loss(loss, degree=degree, suffix=(self.n_iter_,))
                degree = 0

                self.is_converged = self.aggregator.get_converge_status(suffix=(self.n_iter_,))
                LOGGER.info("n_iters: {}, loss: {} converge flag is :{}".format(self.n_iter_, loss, self.is_converged))
                if self.is_converged or self.n_iter_ == max_iter:
                    break
                model_weights = self.model_weights

            batch_num = 0
            for batch_data in batch_data_generator:
                n = batch_data.count()
                # LOGGER.debug("In each batch, lr_weight: {}, batch_data count: {}".format(model_weights.unboxed, n))
                f = functools.partial(self.gradient_operator.compute_gradient,
                                      coef=model_weights.coef_,
                                      intercept=model_weights.intercept_,
                                      fit_intercept=self.fit_intercept)
                grad = batch_data.applyPartitions(f).reduce(fate_operator.reduce_add)
                grad /= n
                # LOGGER.debug('iter: {}, batch_index: {}, grad: {}, n: {}'.format(
                #     self.n_iter_, batch_num, grad, n))

                if self.use_proximal:  # use proximal term
                    model_weights = self.optimizer.update_model(model_weights, grad=grad,
                                                                has_applied=False,
                                                                prev_round_weights=self.prev_round_weights)
                else:
                    model_weights = self.optimizer.update_model(model_weights, grad=grad,
                                                                has_applied=False)

                batch_num += 1
                degree += n

            # validation_strategy.validate(self, self.n_iter_)
            self.callback_list.on_epoch_end(self.n_iter_)
            self.n_iter_ += 1

            if self.stop_training:
                break

        self.set_summary(self.get_model_summary())

    @assert_io_num_rows_equal
    def predict(self, data_instances):

        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)

        data_instances = self.align_data_header(data_instances, self.header)

        LOGGER.info("Start predict is a one_vs_rest task: {}".format(self.need_one_vs_rest))
        if self.need_one_vs_rest:
            predict_result = self.one_vs_rest_obj.predict(data_instances)
            return predict_result

        # predict_wx = self.compute_wx(data_instances, self.model_weights.coef_, self.model_weights.intercept_)
        pred_prob = data_instances.mapValues(lambda v: activation.sigmoid(vec_dot(v.features, self.model_weights.coef_)
                                                                          + self.model_weights.intercept_))

        predict_result = self.predict_score_to_output(data_instances, pred_prob, classes=[0, 1],
                                                      threshold=self.model_param.predict_param.threshold)

        return predict_result
