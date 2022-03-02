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

from federatedml.linear_model.coordinated_linear_model.logistic_regression.base_logistic_regression import \
    BaseLogisticRegression
from federatedml.optim.gradient.hetero_sqn_gradient import sqn_factory
from federatedml.param.logistic_regression_param import HeteroLogisticParam
from federatedml.protobuf.generated import lr_model_meta_pb2
from federatedml.secureprotol import PaillierEncrypt
from federatedml.transfer_variable.transfer_class.hetero_lr_transfer_variable import HeteroLRTransferVariable
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroLRBase(BaseLogisticRegression):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroLogisticRegression'
        self.model_param_name = 'HeteroLogisticRegressionParam'
        self.model_meta_name = 'HeteroLogisticRegressionMeta'
        self.mode = consts.HETERO
        self.aggregator = None
        self.cipher = None
        self.batch_generator = None
        self.gradient_loss_operator = None
        self.converge_procedure = None
        self.model_param = HeteroLogisticParam()
        self.transfer_variable = HeteroLRTransferVariable()

    def _init_model(self, params):
        super()._init_model(params)
        self.encrypted_mode_calculator_param = params.encrypted_mode_calculator_param
        self.cipher_operator = PaillierEncrypt()
        self.cipher.register_paillier_cipher(self.transfer_variable)
        self.converge_procedure.register_convergence(self.transfer_variable)
        self.batch_generator.register_batch_generator(self.transfer_variable)
        self.gradient_loss_operator.register_gradient_procedure(self.transfer_variable)
        # if len(self.component_properties.host_party_idlist) == 1:
        #     LOGGER.debug(f"set_use_async")
        #     self.gradient_loss_operator.set_use_async()
        self.gradient_loss_operator.set_fixed_float_precision(self.model_param.floating_point_precision)

    def _get_meta(self):
        meta_protobuf_obj = lr_model_meta_pb2.LRModelMeta(penalty=self.model_param.penalty,
                                                          tol=self.model_param.tol,
                                                          alpha=self.alpha,
                                                          optimizer=self.model_param.optimizer,
                                                          batch_size=self.batch_size,
                                                          learning_rate=self.model_param.learning_rate,
                                                          max_iter=self.max_iter,
                                                          early_stop=self.model_param.early_stop,
                                                          fit_intercept=self.fit_intercept,
                                                          need_one_vs_rest=self.need_one_vs_rest)
        return meta_protobuf_obj

    def get_model_summary(self):
        header = self.header
        if header is None:
            return {}
        weight_dict, intercept_ = self.get_weight_intercept_dict(header)
        # best_iteration = -1 if self.validation_strategy is None else self.validation_strategy.best_iteration

        summary = {"coef": weight_dict,
                   "intercept": intercept_,
                   "is_converged": self.is_converged,
                   "one_vs_rest": self.need_one_vs_rest,
                   "best_iteration": self.callback_variables.best_iteration}

        if self.callback_variables.validation_summary is not None:
            summary["validation_metrics"] = self.callback_variables.validation_summary
        # if self.validation_strategy:
        #     validation_summary = self.validation_strategy.summary()
        #     if validation_summary:
        #         summary["validation_metrics"] = validation_summary
        return summary
