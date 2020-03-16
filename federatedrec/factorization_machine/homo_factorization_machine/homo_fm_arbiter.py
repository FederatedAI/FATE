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
from federatedrec.factorization_machine.fm_model_weight import FactorizationMachineWeights
from federatedrec.factorization_machine.homo_factorization_machine.homo_fm_base import HomoFMBase
from federatedml.optim import activation
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HomoFMArbiter(HomoFMBase):
    def __init__(self):
        super(HomoFMArbiter, self).__init__()
        # self.re_encrypt_times = []  # Record the times needed for each host

        self.loss_history = []
        self.is_converged = False
        self.role = consts.ARBITER
        self.aggregator = aggregator.Arbiter()
        self.model_weights = None
        self.host_predict_results = []

    def _init_model(self, params):
        super()._init_model(params)

    def fit(self, data_instances=None, validate_data=None):
        validation_strategy = self.init_validation_strategy()

        model_shape = -1
        embed_size = self.init_param_obj.embed_size

        while self.n_iter_ < self.max_iter+1:
            suffix = (self.n_iter_,)

            if (self.n_iter_ > 0 and self.n_iter_ % self.aggregate_iters == 0) or self.n_iter_ == self.max_iter:
                merged_model = self.aggregator.aggregate_and_broadcast(ciphers_dict=None,
                                                                       suffix=suffix)

                if model_shape == -1:
                    if self.init_param_obj.fit_intercept:
                        model_shape = int((len(merged_model._weights) - 1) / (embed_size + 1))
                    else:
                        model_shape = int(len(merged_model._weights) / (embed_size + 1))

                    # Initialize the model
                    fit_intercept = False
                    if self.init_param_obj.fit_intercept:
                        fit_intercept = True
                        self.init_param_obj.fit_intercept = False
                    w_ = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
                    embed_ = self.initializer.init_model([model_shape, self.init_param_obj.embed_size],
                                                         init_params=self.init_param_obj)
                    self.model_weights = \
                        FactorizationMachineWeights(w_, embed_, fit_intercept=fit_intercept)

                total_loss = self.aggregator.aggregate_loss(suffix=suffix)
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
                # self.model_weights = FactorizationMachineWeights(merged_model.coef_, merged_model.embed_,
                #                                                  self.model_param.init_param.fit_intercept)
                merged_model._weights = np.array(merged_model._weights)
                self.model_weights.update(merged_model)
                if self.header is None:
                    self.header = ['x' + str(i) for i in
                                   range(model_shape)]

            validation_strategy.validate(self, self.n_iter_)
            self.n_iter_ += 1

        LOGGER.info("Finish Training task, total iters: {}".format(self.n_iter_))

    def predict(self, data_instantces=None):
        LOGGER.info(f'Start predict task')
