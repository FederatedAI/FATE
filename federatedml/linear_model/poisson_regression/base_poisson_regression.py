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
from google.protobuf import json_format

from arch.api.utils import log_utils
from federatedml.linear_model.linear_model_base import BaseLinearModel
from federatedml.linear_model.linear_model_weight import LinearModelWeights as PoissonRegressionWeights
from federatedml.param.poisson_regression_param import PoissonParam
from federatedml.protobuf.generated import poisson_model_meta_pb2, poisson_model_param_pb2
from federatedml.secureprotol import PaillierEncrypt
from federatedml.param.evaluation_param import EvaluateParam

LOGGER = log_utils.getLogger()


class BasePoissonRegression(BaseLinearModel):
    def __init__(self):
        super(BasePoissonRegression, self).__init__()
        self.model_param = PoissonParam()
        # attribute:
        self.model_name = 'PoissonRegression'
        self.model_param_name = 'PoissonRegressionParam'
        self.model_meta_name = 'PoissonRegressionMeta'
        self.cipher_operator = PaillierEncrypt()

    def _init_model(self, params):
        super()._init_model(params)
        self.encrypted_mode_calculator_param = params.encrypted_mode_calculator_param
        self.exposure_colname = params.exposure_colname

    def get_exposure_index(self, header, exposure_colname):
        try:
            exposure_index = header.index(exposure_colname)
        except:
            exposure_index = -1
        return exposure_index

    def load_instance(self, data_instance):
        """
        return data_instance without exposure
        Parameters
        ----------
        data_instance: DTable of Instances, input data
        """
        if self.exposure_index == -1:
            return data_instance
        if self.exposure_index >= len(data_instance.features):
            raise ValueError(
                "exposure_index {} out of features' range".format(self.exposure_index))
        data_instance.features = np.delete(data_instance.features, self.exposure_index)
        return data_instance

    def load_exposure(self, data_instance):
        """
        return exposure of a given data_instance
        Parameters
        ----------
        data_instance: DTable of Instances, input data
        """
        if self.exposure_index == -1:
            exposure = 1
        else:
            exposure = data_instance.features[self.exposure_index]
        return exposure

    def compute_mu(self, data_instances, coef_, intercept_=0, exposure=None):
        if exposure is None:
            mu = data_instances.mapValues(
                lambda v: np.exp(np.dot(v.features, coef_) + intercept_))
        else:
            mu = data_instances.join(exposure,
                                     lambda v, ei: np.exp(np.dot(v.features, coef_) + intercept_) / ei)
        return mu

    def safe_log(self, v):
        if v == 0:
            return np.log(1e-7)
        return np.log(v)

    def _get_meta(self):
        meta_protobuf_obj = poisson_model_meta_pb2.PoissonModelMeta(
            penalty=self.model_param.penalty,
            eps=self.model_param.eps,
            alpha=self.alpha,
            optimizer=self.model_param.optimizer,
            party_weight=self.model_param.party_weight,
            batch_size=self.batch_size,
            learning_rate=self.model_param.learning_rate,
            max_iter=self.max_iter,
            converge_func=self.model_param.converge_func,
            fit_intercept=self.fit_intercept,
            exposure_colname=self.exposure_colname)
        return meta_protobuf_obj

    def _get_param(self):
        header = self.header
        LOGGER.debug("In get_param, header: {}".format(header))
        if header is None:
            param_protobuf_obj = poisson_model_param_pb2.PoissonModelParam()
            return param_protobuf_obj

        weight_dict = {}
        for idx, header_name in enumerate(header):
            coef_i = self.model_weights.coef_[idx]
            weight_dict[header_name] = coef_i
        intercept_ = self.model_weights.intercept_
        param_protobuf_obj = poisson_model_param_pb2.PoissonModelParam(iters=self.n_iter_,
                                                                       loss_history=self.loss_history,
                                                                       is_converged=self.is_converged,
                                                                       weight=weight_dict,
                                                                       intercept=intercept_,
                                                                       header=header)
        json_result = json_format.MessageToJson(param_protobuf_obj)
        LOGGER.debug("json_result: {}".format(json_result))
        return param_protobuf_obj

    def _load_model(self, model_dict):
        # LOGGER.debug("In load model, model_dict: {}".format(model_dict))
        result_obj = list(model_dict.get('model').values())[0].get(
            self.model_param_name)
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        fit_intercept = meta_obj.fit_intercept
        self.exposure_index = meta_obj.exposure_index

        self.header = list(result_obj.header)
        # LOGGER.debug("In load model, header: {}".format(self.header))
        # For poisson regression arbiter predict function
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
        self.model_weights = PoissonRegressionWeights(l=tmp_vars, fit_intercept=fit_intercept)
    
    def get_metrics_param(self):
        return EvaluateParam(eval_type="regression")

