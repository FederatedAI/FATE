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

import numpy as np

from federatedml.framework.homo.procedure import aggregator
from federatedml.framework.homo.procedure import paillier_cipher
from federatedml.linear_model.linear_model_weight import LinearModelWeights as LogisticRegressionWeights
from federatedml.linear_model.coordinated_linear_model.\
    logistic_regression.homo_logistic_regression.homo_lr_base import HomoLRBase
from federatedml.optim import activation

from federatedml.util import LOGGER
from federatedml.util import consts


class HomoLRArbiter(HomoLRBase):
    def __init__(self):
        super(HomoLRArbiter, self).__init__()
        self.re_encrypt_times = []  # Record the times needed for each host

        self.loss_history = []
        self.is_converged = False
        self.role = consts.ARBITER
        # self.aggregator = aggregator.Arbiter()
        self.model_weights = None
        self.cipher = paillier_cipher.Arbiter()
        self.host_predict_results = []

    def _init_model(self, params):
        super()._init_model(params)
        self.cipher.register_paillier_cipher(self.transfer_variable)

    def fit_binary(self, data_instances=None, validate_data=None):
        self.aggregator = aggregator.Arbiter()
        self.aggregator.register_aggregator(self.transfer_variable)

        # self._server_check_data()

        host_ciphers = self.cipher.paillier_keygen(key_length=self.model_param.encrypt_param.key_length,
                                                   suffix=('fit',))
        host_has_no_cipher_ids = [idx for idx, cipher in host_ciphers.items() if cipher is None]
        self.re_encrypt_times = self.cipher.set_re_cipher_time(host_ciphers)
        max_iter = self.max_iter
        # validation_strategy = self.init_validation_strategy()
        self.callback_list.on_train_begin(data_instances, validate_data)

        if self.component_properties.is_warm_start:
            self.callback_warm_start_init_iter(self.n_iter_)

        while self.n_iter_ < max_iter + 1:
            suffix = (self.n_iter_,)
            self.callback_list.on_epoch_begin(self.n_iter_)

            if ((self.n_iter_ + 1) % self.aggregate_iters == 0) or self.n_iter_ == max_iter:
                merged_model = self.aggregator.aggregate_and_broadcast(ciphers_dict=host_ciphers,
                                                                       suffix=suffix)
                total_loss = self.aggregator.aggregate_loss(host_has_no_cipher_ids, suffix)
                if self.need_call_back_loss:
                    self.callback_loss(self.n_iter_, total_loss)
                self.loss_history.append(total_loss)
                if self.use_loss:
                    converge_var = total_loss
                else:
                    converge_var = np.array(merged_model.unboxed)

                self.is_converged = self.aggregator.send_converge_status(self.converge_func.is_converge,
                                                                         (converge_var,),
                                                                         suffix=(self.n_iter_,))
                LOGGER.info("n_iters: {}, total_loss: {}, converge flag is :{}".format(self.n_iter_,
                                                                                       total_loss,
                                                                                       self.is_converged))

                self.model_weights = LogisticRegressionWeights(merged_model.unboxed,
                                                               self.model_param.init_param.fit_intercept)
                if self.header is None:
                    self.header = ['x' + str(i) for i in range(len(self.model_weights.coef_))]

                if self.is_converged or self.n_iter_ == max_iter:
                    break

            self.cipher.re_cipher(iter_num=self.n_iter_,
                                  re_encrypt_times=self.re_encrypt_times,
                                  host_ciphers_dict=host_ciphers,
                                  re_encrypt_batches=self.re_encrypt_batches)

            # validation_strategy.validate(self, self.n_iter_)
            self.callback_list.on_epoch_end(self.n_iter_)
            self.n_iter_ += 1
            if self.stop_training:
                break

        LOGGER.info("Finish Training task, total iters: {}".format(self.n_iter_))

    def predict(self, data_instantces=None):
        LOGGER.info(f'Start predict task')
        current_suffix = ('predict',)

        LOGGER.info("Start predict is a one_vs_rest task: {}".format(self.need_one_vs_rest))
        if self.need_one_vs_rest:
            predict_result = self.one_vs_rest_obj.predict(data_instantces)
            return predict_result
        host_ciphers = self.cipher.paillier_keygen(key_length=self.model_param.encrypt_param.key_length,
                                                   suffix=current_suffix)
        # LOGGER.debug("Loaded arbiter model: {}".format(self.model_weights.unboxed))
        for idx, cipher in host_ciphers.items():
            if cipher is None:
                continue
            encrypted_model_weights = self.model_weights.encrypted(cipher, inplace=False)
            self.transfer_variable.aggregated_model.remote(obj=encrypted_model_weights.for_remote(),
                                                           role=consts.HOST,
                                                           idx=idx,
                                                           suffix=current_suffix)

        # Receive wx results

        for idx, cipher in host_ciphers.items():
            if cipher is None:
                continue
            encrypted_predict_wx = self.transfer_variable.predict_wx.get(idx=idx, suffix=current_suffix)
            predict_wx = cipher.distribute_decrypt(encrypted_predict_wx)

            prob_table = predict_wx.mapValues(lambda x: activation.sigmoid(x))
            predict_table = prob_table.mapValues(lambda x: 1 if x > self.model_param.predict_param.threshold else 0)

            self.transfer_variable.predict_result.remote(predict_table,
                                                         role=consts.HOST,
                                                         idx=idx,
                                                         suffix=current_suffix)
            self.host_predict_results.append((prob_table, predict_table))
