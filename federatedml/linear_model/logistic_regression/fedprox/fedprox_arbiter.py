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

from arch.api.utils import log_utils
from federatedml.framework.homo.procedure import aggregator
from federatedml.framework.homo.procedure import paillier_cipher
from federatedml.linear_model.linear_model_weight import LinearModelWeights as LogisticRegressionWeights
from federatedml.linear_model.logistic_regression.fedprox.fedprox_base import FedProxBase
from federatedml.optim import activation
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class fedproxArbiter(FedProxBase):
    def __init__(self):
        super(fedproxArbiter, self).__init__()
        self.re_encrypt_times = []  # Record the times needed for each host

        self.loss_history = []
        self.is_converged = False
        self.role = consts.ARBITER
        self.aggregator = aggregator.Arbiter()
        self.model_weights = None
        self.cipher = paillier_cipher.Arbiter()
        self.host_predict_results = []

    def _init_model(self, params):
        super()._init_model(params)
        self.cipher.register_paillier_cipher(self.transfer_variable)

    def fit(self, data_instances=None, validate_data=None):
        
        """This function facilitates the aggregation and transferring of the model weights and losses.

        Args:
            data_instances (pd.DataFrame): the actual data
            validate_data (str): if validate_data is not None, then validation strategy is used to check
            the model performance

        Returns: 
            update model_weights, losses and send them to guest and host


        """

        #Obtain host public key
        host_ciphers = self.cipher.paillier_keygen(key_length=self.model_param.encrypt_param.key_length,
                                                   suffix=('fit',))
        host_has_no_cipher_ids = [idx for idx, cipher in host_ciphers.items() if cipher is None]
        self.re_encrypt_times = self.cipher.set_re_cipher_time(host_ciphers)

        #Set the elements required for the loop
        max_iter = self.max_iter

        #Initialize the validation to check model performance
        validation_strategy = self.init_validation_strategy()

        self.guest_size = self.transfer_variable.guest_size.get(idx = -1) # getting guest size
        self.host_size = self.transfer_variable.host_size.get(idx = -1) # getting host size
        LOGGER.info("**AIFEL** guest_size: {}, host_size: {}".format(self.guest_size, self.host_size))

        #This loop is aggregating the models and losses and sending them back to guest and host
        while self.n_iter_ < max_iter:
            suffix = (self.n_iter_,)

            if self.n_iter_ > 0 and self.n_iter_ % self.aggregate_iters == 0:
                merged_model = self.aggregator.aggregate_and_broadcast(ciphers_dict=host_ciphers,
                                                                       suffix=suffix)
                total_loss = self.aggregator.aggregate_loss(host_has_no_cipher_ids, suffix)
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
                if self.is_converged:
                    break
                self.model_weights = LogisticRegressionWeights(merged_model.unboxed,
                                                               self.model_param.init_param.fit_intercept)
                if self.header is None:
                    self.header = ['x' + str(i) for i in range(len(self.model_weights.coef_))]

            self.cipher.re_cipher(iter_num=self.n_iter_,
                                  re_encrypt_times=self.re_encrypt_times,
                                  host_ciphers_dict=host_ciphers,
                                  re_encrypt_batches=self.re_encrypt_batches)
            
            validation_strategy.validate(self, self.n_iter_)
            self.n_iter_ += 1

        LOGGER.info("**AIFEL** Finish Training task, total iters: {}".format(self.n_iter_))

    def predict(self, data_instantces=None):
        LOGGER.info(f'**AIFEL** Start predict task')
        current_suffix = ('predict',)
        host_ciphers = self.cipher.paillier_keygen(key_length=self.model_param.encrypt_param.key_length,
                                                   suffix=current_suffix)

        LOGGER.debug("Loaded arbiter model: {}".format(self.model_weights.unboxed))
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

    # def run(self, component_parameters=None, args=None):
    #     self._init_runtime_parameters(component_parameters)
    #     data_sets = args["data"]
    #
    #     data_statement_dict = list(data_sets.values())[0]
    #     need_eval = False
    #     for data_key in data_sets:
    #         if 'eval_data' in data_sets[data_key]:
    #             need_eval = True
    #
    #     LOGGER.debug("data_sets: {}, data_statement_dict: {}".format(data_sets, data_statement_dict))
    #     if self.need_cv:
    #         LOGGER.info("Task is cross validation.")
    #         self.cross_validation(None)
    #         return
    #
    #     elif not "model" in args:
    #         LOGGER.info("Task is fit")
    #         self.set_flowid('fit')
    #         self.fit()
    #         self.set_flowid('predict')
    #         self.predict()
    #         if need_eval:
    #             self.set_flowid('validate')
    #             self.predict()
    #     else:
    #         LOGGER.info("Task is predict")
    #         self.load_model(args)
    #         self.set_flowid('predict')
    #         self.predict()
