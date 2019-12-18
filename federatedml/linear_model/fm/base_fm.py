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
from federatedml.linear_model.linear_model_weight import LinearModelWeights as FMWeights   # modify
from federatedml.optim.initialize import Initializer
from federatedml.protobuf.generated import fm_model_param_pb2
from federatedml.one_vs_rest.one_vs_rest import one_vs_rest_factory
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class BaseFM(BaseLinearModel):
    def __init__(self):
        super(BaseFM, self).__init__()
        # attribute:

        self.initializer = Initializer()
        self.model_name = 'FM'
        self.model_param_name = 'FMParam'
        self.model_meta_name = 'FMMeta'

        # one_ve_rest parameter
        self.need_one_vs_rest = None
        self.one_vs_rest_classes = []
        self.one_vs_rest_obj = None

    def _init_model(self, params):
        super()._init_model(params)
        self.one_vs_rest_obj = one_vs_rest_factory(self, role=self.role, mode=self.mode, has_arbiter=True)

    def get_single_model_param(self):
        weight_dict = {}
        v_dict = {}
        LOGGER.debug("in get_single_model_param, model_weights: {}, coef: {}, header: {}".format(
            self.model_weights.unboxed, self.model_weights.coef_, self.header
        ))
        LOGGER.debug("in get_single_model_param, model_feature_emb: {}, coef: {}, header: {}".format(
            self.model_feature_emb.unboxed, self.model_feature_emb.coef_, self.header
        ))

        for idx, header_name in enumerate(self.header):
            coef_i = self.model_weights.coef_[idx]
            weight_dict[header_name] = coef_i
        for idx, header_name in enumerate(self.header):
            emb = {}
            emb['item'] = self.model_feature_emb.coef_[idx].tolist()
            coef_i = fm_model_param_pb2.DoubleArray(**emb)
            v_dict[header_name] = coef_i

        result = {'iters': self.n_iter_,
                  'loss_history': self.loss_history,
                  'is_converged': self.is_converged,
                  'weight': weight_dict,
                  'v':v_dict,
                  'intercept': self.model_weights.intercept_,
                  'header': self.header
                  }
        return result

    def _get_param(self):
        header = self.header
        LOGGER.debug("In get_param, header: {}".format(header))
        if header is None:
            param_protobuf_obj = fm_model_param_pb2.FMModelParam()
            return param_protobuf_obj
        if self.need_one_vs_rest:
            # one_vs_rest_class = list(map(str, self.one_vs_rest_obj.classes))
            one_vs_rest_result = self.one_vs_rest_obj.save(fm_model_param_pb2.SingleModel)
            single_result = {'header': header, 'need_one_vs_rest': True}
        else:
            one_vs_rest_result = None
            single_result = self.get_single_model_param()
            single_result['need_one_vs_rest'] = False
        single_result['one_vs_rest_result'] = one_vs_rest_result
        LOGGER.debug("in _get_param, single_result: {}".format(single_result))

        param_protobuf_obj = fm_model_param_pb2.FMModelParam(**single_result)
        json_result = json_format.MessageToJson(param_protobuf_obj)
        LOGGER.debug("json_result: {}".format(json_result))
        return param_protobuf_obj

    def _load_model(self, model_dict):
        LOGGER.debug("Start Loading model")
        result_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        self.fit_intercept = meta_obj.fit_intercept
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
        tmp_vars_w = np.zeros(feature_shape)
        tmp_vars_v = np.zeros(feature_shape)

        weight_dict = dict(single_model_obj.weight)
        v_dict = dict(single_model_obj.v)

        for idx, header_name in enumerate(self.header):
            tmp_vars_w[idx] = weight_dict.get(header_name)
        if self.fit_intercept:
            tmp_vars_w = np.append(tmp_vars_w, single_model_obj.intercept)
        self.model_weights = FMWeights(tmp_vars_w, fit_intercept=self.fit_intercept)

        for idx, header_name in enumerate(self.header):
            tmp_vars_v[idx] = v_dict.get(header_name)
        self.model_feature_emb = FMWeights(tmp_vars_v, fit_intercept=self.fit_intercept)

        return self


    def one_vs_rest_fit(self, train_data=None, validate_data=None):
        LOGGER.debug("Class num larger than 2, need to do one_vs_rest")
        self.one_vs_rest_obj.fit(data_instances=train_data, validate_data=validate_data)

    def one_vs_rest_predict(self, validate_data):
        if not self.one_vs_rest_obj:
            LOGGER.warning("Not one_vs_rest fit before, return now")
            return
        return self.one_vs_rest_obj.predict(data_instances=validate_data)


