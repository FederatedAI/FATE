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
from federatedml.framework.homo.procedure import paillier_cipher
from federatedml.linear_model.linear_model_weight import LinearModelWeights as LogisticRegressionWeights
from federatedml.linear_model.coordinated_linear_model.logistic_regression.\
    homo_logistic_regression.homo_lr_base import HomoLRBase
from federatedml.model_selection import MiniBatch
from federatedml.feature.instance import Instance
from federatedml.optim import activation
from federatedml.util.fate_operator import vec_dot
from federatedml.optim.gradient.homo_lr_gradient import LogisticGradient, TaylorLogisticGradient
from federatedml.protobuf.generated import lr_model_param_pb2
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.util.io_check import assert_io_num_rows_equal


class HomoLRHost(HomoLRBase):
    def __init__(self):
        super(HomoLRHost, self).__init__()
        self.gradient_operator = None
        self.loss_history = []
        self.is_converged = False
        self.role = consts.HOST
        # self.aggregator = aggregator.Host()
        self.model_weights = None
        self.cipher = None

    def _init_model(self, params):
        super()._init_model(params)
        if self.component_properties.has_arbiter:
            self.cipher = paillier_cipher.Host()
        if params.encrypt_param.method in [consts.PAILLIER]:
            self.cipher.register_paillier_cipher(self.transfer_variable)
            self.use_encrypt = True
            self.gradient_operator = TaylorLogisticGradient()
            self.re_encrypt_batches = params.re_encrypt_batches
        else:
            self.use_encrypt = False
            self.gradient_operator = LogisticGradient()

    def fit_binary(self, data_instances, validate_data=None):
        self.aggregator = aggregator.Host()
        self.aggregator.register_aggregator(self.transfer_variable)

        self._client_check_data(data_instances)
        self.callback_list.on_train_begin(data_instances, validate_data)

        pubkey = self.cipher.gen_paillier_pubkey(enable=self.use_encrypt, suffix=('fit',))
        if self.use_encrypt:
            self.cipher_operator.set_public_key(pubkey)

        if not self.component_properties.is_warm_start:
            self.model_weights = self._init_model_variables(data_instances)
            if self.use_encrypt:
                w = self.cipher_operator.encrypt_list(self.model_weights.unboxed)
            else:
                w = list(self.model_weights.unboxed)
            self.model_weights = LogisticRegressionWeights(w, self.model_weights.fit_intercept)
        else:
            self.callback_warm_start_init_iter(self.n_iter_)

        # LOGGER.debug("After init, model_weights: {}".format(self.model_weights.unboxed))

        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)

        total_batch_num = mini_batch_obj.batch_nums

        if self.use_encrypt:
            re_encrypt_times = (total_batch_num - 1) // self.re_encrypt_batches + 1
            # LOGGER.debug("re_encrypt_times is :{}, batch_size: {}, total_batch_num: {}, re_encrypt_batches: {}".format(
            #     re_encrypt_times, self.batch_size, total_batch_num, self.re_encrypt_batches))
            self.cipher.set_re_cipher_time(re_encrypt_times)

        # total_data_num = data_instances.count()
        # LOGGER.debug("Current data count: {}".format(total_data_num))

        model_weights = self.model_weights
        self.prev_round_weights = copy.deepcopy(model_weights)
        degree = 0
        while self.n_iter_ < self.max_iter + 1:
            self.callback_list.on_epoch_begin(self.n_iter_)

            batch_data_generator = mini_batch_obj.mini_batch_data_generator()
            self.optimizer.set_iters(self.n_iter_)

            self.optimizer.set_iters(self.n_iter_)
            if ((self.n_iter_ + 1) % self.aggregate_iters == 0) or self.n_iter_ == self.max_iter:
                weight = self.aggregator.aggregate_then_get(model_weights, degree=degree,
                                                            suffix=self.n_iter_)
                # LOGGER.debug("Before aggregate: {}, degree: {} after aggregated: {}".format(
                #     model_weights.unboxed / degree,
                #     degree,
                #     weight.unboxed))
                self.model_weights = LogisticRegressionWeights(weight.unboxed, self.fit_intercept)
                if not self.use_encrypt:
                    loss = self._compute_loss(data_instances, self.prev_round_weights)
                    self.aggregator.send_loss(loss, degree=degree, suffix=(self.n_iter_,))
                    LOGGER.info("n_iters: {}, loss: {}".format(self.n_iter_, loss))
                degree = 0
                self.is_converged = self.aggregator.get_converge_status(suffix=(self.n_iter_,))
                LOGGER.info("n_iters: {}, is_converge: {}".format(self.n_iter_, self.is_converged))
                if self.is_converged or self.n_iter_ == self.max_iter:
                    break
                model_weights = self.model_weights

            batch_num = 0
            for batch_data in batch_data_generator:
                n = batch_data.count()
                degree += n
                LOGGER.debug('before compute_gradient')
                f = functools.partial(self.gradient_operator.compute_gradient,
                                      coef=model_weights.coef_,
                                      intercept=model_weights.intercept_,
                                      fit_intercept=self.fit_intercept)
                grad = batch_data.applyPartitions(f).reduce(fate_operator.reduce_add)
                grad /= n
                if self.use_proximal:  # use additional proximal term
                    model_weights = self.optimizer.update_model(model_weights,
                                                                grad=grad,
                                                                has_applied=False,
                                                                prev_round_weights=self.prev_round_weights)
                else:
                    model_weights = self.optimizer.update_model(model_weights,
                                                                grad=grad,
                                                                has_applied=False)

                if self.use_encrypt and batch_num % self.re_encrypt_batches == 0:
                    LOGGER.debug("Before accept re_encrypted_model, batch_iter_num: {}".format(batch_num))
                    w = self.cipher.re_cipher(w=model_weights.unboxed,
                                              iter_num=self.n_iter_,
                                              batch_iter_num=batch_num)
                    model_weights = LogisticRegressionWeights(w, self.fit_intercept)
                batch_num += 1

            # validation_strategy.validate(self, self.n_iter_)
            self.callback_list.on_epoch_end(self.n_iter_)
            self.n_iter_ += 1
            if self.stop_training:
                break

        self.set_summary(self.get_model_summary())
        LOGGER.info("Finish Training task, total iters: {}".format(self.n_iter_))

    def get_model_summary(self):
        header = self.header
        if header is None:
            return {}
        if not self.use_encrypt:
            weight_dict, intercept_ = self.get_weight_intercept_dict(header)
        else:
            weight_dict = {}
            intercept_ = None
        best_iteration = -1 if self.validation_strategy is None else self.validation_strategy.best_iteration

        summary = {"coef": weight_dict,
                   "intercept": intercept_,
                   "is_converged": self.is_converged,
                   "best_iteration": best_iteration}
        return summary

    @assert_io_num_rows_equal
    def predict(self, data_instances):

        LOGGER.info(f'Start predict task')
        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        LOGGER.info("Start predict is a one_vs_rest task: {}".format(self.need_one_vs_rest))
        if self.need_one_vs_rest:
            predict_result = self.one_vs_rest_obj.predict(data_instances)
            return predict_result

        suffix = ('predict',)
        if self.component_properties.has_arbiter:
            pubkey = self.cipher.gen_paillier_pubkey(enable=self.use_encrypt, suffix=suffix)
        else:
            if self.use_encrypt:
                raise ValueError(f"In use_encrypt case, arbiter should be set")
            pubkey = None
        if self.use_encrypt:
            self.cipher_operator.set_public_key(pubkey)

            final_model = self.transfer_variable.aggregated_model.get(idx=0, suffix=suffix)
            model_weights = LogisticRegressionWeights(final_model.unboxed, self.fit_intercept)
            wx = self.compute_wx(data_instances, model_weights.coef_, model_weights.intercept_)
            self.transfer_variable.predict_wx.remote(wx, consts.ARBITER, 0, suffix=suffix)
            predict_result = self.transfer_variable.predict_result.get(idx=0, suffix=suffix)
            # predict_result = predict_result.join(data_instances, lambda p, d: [d.label, p, None,
            #                                                                    {"0": None, "1": None}])
            predict_result = predict_result.join(data_instances, lambda p, d:
                                                 Instance(features=[d.label, p, None, {"0": None, "1": None}],
                                                          inst_id=d.inst_id)
                                                 )
        else:
            pred_prob = data_instances.mapValues(
                lambda v: activation.sigmoid(vec_dot(v.features, self.model_weights.coef_)
                                             + self.model_weights.intercept_))
            predict_result = self.predict_score_to_output(data_instances, pred_prob, classes=[0, 1],
                                                          threshold=self.model_param.predict_param.threshold)

        return predict_result

    def get_single_model_param(self):
        header = self.header

        weight_dict = {}

        if not self.use_encrypt and not self.component_properties.need_cv:
            lr_vars = self.model_weights.coef_
            for idx, header_name in enumerate(header):
                coef_i = lr_vars[idx]
                weight_dict[header_name] = coef_i
            intercept = self.model_weights.intercept_
        else:
            intercept = 0

        result = {'iters': self.n_iter_,
                  'loss_history': self.loss_history,
                  'is_converged': self.is_converged,
                  'weight': weight_dict,
                  'intercept': intercept,
                  'header': self.header,
                  'best_iteration': -1 if self.validation_strategy is None else
                  self.validation_strategy.best_iteration
                  }
        return result
