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
from federatedml.linear_model.linear_model_weight import LinearModelWeights as LogisticRegressionWeights
from federatedml.one_vs_rest.one_vs_rest import one_vs_rest_factory
from federatedml.optim.initialize import Initializer
from federatedml.param.logistic_regression_param import InitParam
from federatedml.protobuf.generated import lr_model_param_pb2
from federatedml.util.fate_operator import vec_dot

LOGGER = log_utils.getLogger()


class BaseLogisticRegression(BaseLinearModel):
    def __init__(self):
        super(BaseLogisticRegression, self).__init__()
        # attribute:

        self.initializer = Initializer()
        self.model_name = 'LogisticRegression'
        self.model_param_name = 'LogisticRegressionParam'
        self.model_meta_name = 'LogisticRegressionMeta'

        # one_ve_rest parameter
        self.need_one_vs_rest = None
        self.one_vs_rest_classes = []
        self.one_vs_rest_obj = None

    def _init_model(self, params):
        super()._init_model(params)
        self.one_vs_rest_obj = one_vs_rest_factory(self, role=self.role, mode=self.mode, has_arbiter=True)

    def compute_wx(self, data_instances, coef_, intercept_=0):
        return data_instances.mapValues(lambda v: vec_dot(v.features, coef_) + intercept_)

    def get_single_model_param(self):
        weight_dict = {}
        LOGGER.debug("in get_single_model_param, model_weights: {}, coef: {}, header: {}".format(
            self.model_weights.unboxed, self.model_weights.coef_, self.header
        ))
        for idx, header_name in enumerate(self.header):
            coef_i = self.model_weights.coef_[idx]
            weight_dict[header_name] = coef_i

        result = {'iters': self.n_iter_,
                  'loss_history': self.loss_history,
                  'is_converged': self.is_converged,
                  'weight': weight_dict,
                  'intercept': self.model_weights.intercept_,
                  'header': self.header
                  }
        return result

    def _get_param(self):
        header = self.header
        LOGGER.debug("In get_param, header: {}".format(header))
        if header is None:
            param_protobuf_obj = lr_model_param_pb2.LRModelParam()
            return param_protobuf_obj
        if self.need_one_vs_rest:
            # one_vs_rest_class = list(map(str, self.one_vs_rest_obj.classes))
            one_vs_rest_result = self.one_vs_rest_obj.save(lr_model_param_pb2.SingleModel)
            single_result = {'header': header, 'need_one_vs_rest': True}
        else:
            one_vs_rest_result = None
            single_result = self.get_single_model_param()
            single_result['need_one_vs_rest'] = False
        single_result['one_vs_rest_result'] = one_vs_rest_result
        LOGGER.debug("in _get_param, single_result: {}".format(single_result))

        param_protobuf_obj = lr_model_param_pb2.LRModelParam(**single_result)
        json_result = json_format.MessageToJson(param_protobuf_obj)
        LOGGER.debug("json_result: {}".format(json_result))
        return param_protobuf_obj

    def load_model(self, model_dict):
        LOGGER.debug("Start Loading model")
        result_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        # self.fit_intercept = meta_obj.fit_intercept
        if self.init_param_obj is None:
            self.init_param_obj = InitParam()
        self.init_param_obj.fit_intercept = meta_obj.fit_intercept
        self.header = list(result_obj.header)
        # For hetero-lr arbiter predict function
        if self.header is None:
            return

        need_one_vs_rest = result_obj.need_one_vs_rest
        LOGGER.debug("in _load_model need_one_vs_rest: {}".format(need_one_vs_rest))
        if need_one_vs_rest:
            one_vs_rest_result = result_obj.one_vs_rest_result
            self.one_vs_rest_obj = one_vs_rest_factory(classifier=self, role=self.role,
                                                       mode=self.mode, has_arbiter=True)
            self.one_vs_rest_obj.load_model(one_vs_rest_result)
            self.need_one_vs_rest = True
        else:
            self.load_single_model(result_obj)
            self.need_one_vs_rest = False

    def load_single_model(self, single_model_obj):
        LOGGER.info("It's a binary task, start to load single model")
        feature_shape = len(self.header)
        tmp_vars = np.zeros(feature_shape)
        weight_dict = dict(single_model_obj.weight)

        for idx, header_name in enumerate(self.header):
            tmp_vars[idx] = weight_dict.get(header_name)

        if self.fit_intercept:
            tmp_vars = np.append(tmp_vars, single_model_obj.intercept)
        self.model_weights = LogisticRegressionWeights(tmp_vars, fit_intercept=self.fit_intercept)
        self.n_iter_ = single_model_obj.iters
        return self

    def one_vs_rest_fit(self, train_data=None, validate_data=None):
        LOGGER.debug("Class num larger than 2, need to do one_vs_rest")
        self.one_vs_rest_obj.fit(data_instances=train_data, validate_data=validate_data)

