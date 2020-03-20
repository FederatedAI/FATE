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
#

import numpy as np

from arch.api.utils import log_utils
from federatedml.linear_model.linear_model_weight import LinearModelWeights as LinearRegressionWeights
from federatedml.linear_model.linear_model_base import BaseLinearModel
from federatedml.optim.initialize import Initializer
from federatedml.param.linear_regression_param import LinearParam
from federatedml.protobuf.generated import linr_model_param_pb2, linr_model_meta_pb2
from federatedml.secureprotol import PaillierEncrypt
from federatedml.param.evaluation_param import EvaluateParam
from federatedml.util.fate_operator import vec_dot

LOGGER = log_utils.getLogger()


class BaseLinearRegression(BaseLinearModel):
    def __init__(self):
        super(BaseLinearRegression, self).__init__()
        self.model_param = LinearParam()
        # attribute:
        self.n_iter_ = 0
        self.feature_shape = None

        self.gradient_operator = None
        self.initializer = Initializer()
        self.transfer_variable = None
        self.loss_history = []
        self.is_converged = False
        self.header = None
        self.model_name = 'LinearRegression'
        self.model_param_name = 'LinearRegressionParam'
        self.model_meta_name = 'LinearRegressionMeta'
        self.role = ''
        self.mode = ''
        self.schema = {}
        self.cipher_operator = PaillierEncrypt()

    def _init_model(self, params):
        super()._init_model(params)
        self.encrypted_mode_calculator_param = params.encrypted_mode_calculator_param

    def compute_wx(self, data_instances, coef_, intercept_=0):
        return data_instances.mapValues(
            lambda v: vec_dot(v.features, coef_) + intercept_)

    def _get_meta(self):
        meta_protobuf_obj = linr_model_meta_pb2.LinRModelMeta(penalty=self.model_param.penalty,
                                                              tol=self.model_param.tol,
                                                              alpha=self.alpha,
                                                              optimizer=self.model_param.optimizer,
                                                              batch_size=self.batch_size,
                                                              learning_rate=self.model_param.learning_rate,
                                                              max_iter=self.max_iter,
                                                              early_stop=self.model_param.early_stop,
                                                              fit_intercept=self.fit_intercept)
        return meta_protobuf_obj

    def _get_param(self):
        header = self.header
        LOGGER.debug("In get_param, header: {}".format(header))
        if header is None:
            param_protobuf_obj = linr_model_param_pb2.LinRModelParam()
            return param_protobuf_obj

        weight_dict = {}
        for idx, header_name in enumerate(header):
            coef_i = self.model_weights.coef_[idx]
            weight_dict[header_name] = coef_i
        intercept_ = self.model_weights.intercept_
        param_protobuf_obj = linr_model_param_pb2.LinRModelParam(iters=self.n_iter_,
                                                                 loss_history=self.loss_history,
                                                                 is_converged=self.is_converged,
                                                                 weight=weight_dict,
                                                                 intercept=intercept_,
                                                                 header=header)
        return param_protobuf_obj

    def load_model(self, model_dict):
        result_obj = list(model_dict.get('model').values())[0].get(
            self.model_param_name)

        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        fit_intercept = meta_obj.fit_intercept

        self.header = list(result_obj.header)
        if self.header is None:
            return

        feature_shape = len(self.header)
        tmp_vars = np.zeros(feature_shape)
        weight_dict = dict(result_obj.weight)
        self.intercept_ = result_obj.intercept

        for idx, header_name in enumerate(self.header):
            tmp_vars[idx] = weight_dict.get(header_name)

        if fit_intercept:
            tmp_vars = np.append(tmp_vars, result_obj.intercept)
        self.model_weights = LinearRegressionWeights(l=tmp_vars, fit_intercept=fit_intercept)
        self.n_iter_ = result_obj.iters
    
    def get_metrics_param(self):
        return EvaluateParam(eval_type="regression")
