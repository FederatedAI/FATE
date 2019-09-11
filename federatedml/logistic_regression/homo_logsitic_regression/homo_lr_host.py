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

from arch.api.proto import lr_model_param_pb2
from arch.api.utils import log_utils
from federatedml.framework.homo.procedure import aggregator, predict_procedure
from federatedml.framework.homo.procedure import paillier_cipher
from federatedml.logistic_regression.homo_logsitic_regression.homo_lr_base import HomoLRBase
from federatedml.logistic_regression.logistic_regression_variables import LogisticRegressionWeights
from federatedml.model_selection import MiniBatch
from federatedml.optim.gradient.logistic_gradient import LogisticGradient, TaylorLogisticGradient
from federatedml.util import consts
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


class HomoLRHost(HomoLRBase):
    def __init__(self):
        super(HomoLRHost, self).__init__()
        self.gradient_operator = None
        self.loss_history = []
        self.is_converged = False
        self.role = consts.HOST
        self.aggregator = aggregator.Host()
        self.lr_variables = None
        self.cipher = paillier_cipher.Host()
        self.predict_procedure = predict_procedure.Host()

    def _init_model(self, params):
        super()._init_model(params)
        self.cipher.register_paillier_cipher(self.transfer_variable)
        if params.encrypt_param.method in [consts.PAILLIER]:
            self.use_encrypt = True
            self.gradient_operator = TaylorLogisticGradient()
            self.re_encrypt_batches = params.re_encrypt_batches
        else:
            self.use_encrypt = False
            self.gradient_operator = LogisticGradient()
        self.predict_procedure.register_predict_sync(self.transfer_variable, self)

    def fit(self, data_instances):
        LOGGER.debug("Start data count: {}".format(data_instances.count()))

        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)

        pubkey = self.cipher.gen_paillier_pubkey(enable=self.use_encrypt, suffix=('fit',))
        if self.use_encrypt:
            self.cipher_operator.set_public_key(pubkey)

        self.lr_variables = self._init_model_variables(data_instances)
        w = self.lr_variables.parameters
        w = self.cipher_operator.encrypt_list(w)
        self.lr_variables = LogisticRegressionWeights(w, self.lr_variables.fit_intercept)

        LOGGER.debug("After init, lr_variable params: {}".format(self.lr_variables.parameters))

        # self.lr_variables = self.lr_variables.encrypted(cipher=self.cipher_operator)

        max_iter = self.max_iter

        LOGGER.debug("Current data count: {}".format(data_instances.count()))
        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)

        total_batch_num = mini_batch_obj.batch_nums

        if self.use_encrypt:
            re_encrypt_times = total_batch_num // self.re_encrypt_batches
            LOGGER.debug("re_encrypt_times is :{}, batch_size: {}, total_batch_num: {}, re_encrypt_batches: {}".format(
                re_encrypt_times, self.batch_size, total_batch_num, self.re_encrypt_batches))
            self.cipher.set_re_cipher_time(re_encrypt_times)

        while self.n_iter_ < max_iter:
            batch_data_generator = mini_batch_obj.mini_batch_data_generator()

            batch_num = 0
            iter_loss = 0
            for batch_data in batch_data_generator:
                n = batch_data.count()
                f = functools.partial(self.gradient_operator.compute,
                                      coef=self.lr_variables.coef_,
                                      intercept=self.lr_variables.intercept_,
                                      fit_intercept=self.fit_intercept)
                grad_loss = batch_data.mapPartitions(f)
                grad, loss = grad_loss.reduce(fate_operator.reduce_add)
                grad /= n
                self.lr_variables = self.optimizer.update_model(self.lr_variables, grad, has_applied=False)
                if not self.use_encrypt:
                    loss /= n
                    loss_norm = self.optimizer.loss_norm(self.lr_variables)
                    iter_loss += loss
                    if loss_norm is not None:
                        iter_loss += loss_norm

                LOGGER.debug('iter: {}, batch_index: {}, grad: {}, loss: {}, n: {}, iter_loss :{}'.format(
                    self.n_iter_, batch_num,
                    grad, loss, n, iter_loss))

                batch_num += 1
                if self.use_encrypt and batch_num % self.re_encrypt_batches == 0:
                    w = self.cipher.re_cipher(w=self.lr_variables.parameters,
                                              iter_num=self.n_iter_,
                                              batch_iter_num=batch_num)
                    self.lr_variables = LogisticRegressionWeights(w, self.fit_intercept)

            LOGGER.debug("Before aggregate, lr_variable params: {}".format(self.lr_variables.parameters))
            self.aggregator.send_model_for_aggregate(self.lr_variables, self.n_iter_)
            if not self.use_encrypt:
                iter_loss /= batch_num
                self.callback_loss(self.n_iter_, iter_loss)
                self.loss_history.append(iter_loss)
                self.aggregator.send_loss(iter_loss, self.n_iter_)
            weight = self.aggregator.get_aggregated_model(self.n_iter_)
            self.lr_variables = LogisticRegressionWeights(weight.parameters, self.fit_intercept)
            self.is_converged = self.aggregator.get_converge_status(suffix=(self.n_iter_,))
            LOGGER.info("n_iters: {}, converge flag is :{}".format(self.n_iter_, self.is_converged))
            if self.is_converged:
                break
            self.n_iter_ += 1

    def predict(self, data_instances):
        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)
        suffix = ('predict',)
        pubkey = self.cipher.gen_paillier_pubkey(enable=self.use_encrypt, suffix=suffix)
        if self.use_encrypt:
            self.cipher_operator.set_public_key(pubkey)

        predict_result = self.predict_procedure.start_predict(data_instances,
                                                              self.lr_variables,
                                                              self.model_param.predict_param.threshold,
                                                              self.use_encrypt,
                                                              self.fit_intercept,
                                                              suffix=suffix)
        return predict_result

    def _get_param(self):
        if self.need_one_vs_rest:
            one_vs_rest_class = list(map(str, self.one_vs_rest_obj.classes))
            param_protobuf_obj = lr_model_param_pb2.LRModelParam(iters=self.n_iter_,
                                                                 loss_history=self.loss_history,
                                                                 is_converged=self.is_converged,
                                                                 weight={},
                                                                 intercept=0,
                                                                 need_one_vs_rest=self.need_one_vs_rest,
                                                                 one_vs_rest_classes=one_vs_rest_class)
            return param_protobuf_obj

        header = self.header

        weight_dict = {}
        intercept = 0
        if not self.use_encrypt:
            lr_vars = self.lr_variables.coef_
            for idx, header_name in enumerate(header):
                coef_i = lr_vars[idx]
                weight_dict[header_name] = coef_i
            intercept = self.lr_variables.intercept_

        param_protobuf_obj = lr_model_param_pb2.LRModelParam(iters=self.n_iter_,
                                                             loss_history=self.loss_history,
                                                             is_converged=self.is_converged,
                                                             weight=weight_dict,
                                                             intercept=intercept,
                                                             need_one_vs_rest=self.need_one_vs_rest,
                                                             header=header)
        from google.protobuf import json_format
        json_result = json_format.MessageToJson(param_protobuf_obj)
        LOGGER.debug("json_result: {}".format(json_result))
        return param_protobuf_obj
