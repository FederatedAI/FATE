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
from federatedml.framework.homo.procedure import aggregator, predict_procedure
from federatedml.framework.homo.procedure import paillier_cipher
from federatedml.logistic_regression.homo_logsitic_regression.homo_lr_base import HomoLRBase
from federatedml.logistic_regression.logistic_regression_variables import LogisticRegressionVariables
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HomoLRArbiter(HomoLRBase):
    def __init__(self):
        super(HomoLRArbiter, self).__init__()
        self.re_encrypt_times = []  # Record the times needed for each host

        self.loss_history = []
        self.is_converged = False
        self.role = consts.ARBITER
        self.aggregator = aggregator.Arbiter()
        self.lr_variables = None
        self.cipher = paillier_cipher.Arbiter()
        self.predict_procedure = predict_procedure.Arbiter()

    def _init_model(self, params):
        super()._init_model(params)
        self.cipher.register_paillier_cipher(self.transfer_variable)
        self.converge_flag_transfer = self.transfer_variable.converge_flag
        self.predict_procedure.register_predict_sync(self.transfer_variable)

    def fit(self, data_instances):
        host_ciphers = self.cipher.paillier_keygen(key_length=self.model_param.encrypt_param.key_length,
                                                   suffix=('fit',))
        host_has_no_cipher_ids = [idx for idx, cipher in host_ciphers.items() if cipher is None]
        self.re_encrypt_times = self.cipher.set_re_cipher_time(host_ciphers)
        max_iter = self.max_iter

        while self.n_iter_ < max_iter:
            self.cipher.re_cipher(iter_num=self.n_iter_,
                                  re_encrypt_times=self.re_encrypt_times,
                                  host_ciphers_dict=host_ciphers,
                                  re_encrypt_batches=self.re_encrypt_batches)
            suffix = (self.n_iter_,)

            merged_model = self.aggregator.aggregate_and_broadcast(ciphers_dict=host_ciphers,
                                                                   suffix=suffix)
            self.lr_variables = LogisticRegressionVariables(merged_model.for_remote().parameters,
                                                            self.model_param.init_param.fit_intercept)
            total_loss = self.aggregator.aggregate_loss(idx=host_has_no_cipher_ids,
                                                        suffix=suffix)
            self.callback_loss(self.n_iter_, total_loss)
            self.loss_history.append(total_loss)
            if self.use_loss:
                converge_var = total_loss
            else:
                converge_var = merged_model
            self.aggregator.check_converge_status(self.converge_func.is_converge,
                                                  (converge_var,),
                                                  suffix=(self.n_iter_,))

            LOGGER.info("n_iters: {}, total_loss: {}, converge flag is :{}".format(self.n_iter_,
                                                                                   total_loss,
                                                                                   self.is_converged))
            if self.is_converged:
                break
            self.n_iter_ += 1

    def predict(self, data_instantces):
        current_suffix = ('predict',)

        host_ciphers = self.cipher.paillier_keygen(key_length=self.model_param.encrypt_param.key_length,
                                                   suffix=current_suffix)
        self.predict_procedure.start_predict(host_ciphers,
                                             self.lr_variables,
                                             self.model_param.predict_param.threshold,
                                             current_suffix)


