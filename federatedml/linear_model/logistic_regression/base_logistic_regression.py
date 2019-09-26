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
import copy

import numpy as np
from google.protobuf import json_format

from arch.api.utils import log_utils
from federatedml.linear_model.linear_model_base import BaseLinearModel
from federatedml.linear_model.linear_model_weight import LinearModelWeights as LogisticRegressionWeights
from federatedml.one_vs_rest.one_vs_rest import OneVsRest
from federatedml.optim.initialize import Initializer
from federatedml.protobuf.generated import lr_model_param_pb2
from federatedml.util import consts

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
        self.need_one_vs_rest = False
        self.one_vs_rest_classes = []
        self.one_vs_rest_obj = None

    def compute_wx(self, data_instances, coef_, intercept_=0):
        return data_instances.mapValues(lambda v: np.dot(v.features, coef_) + intercept_)

    def _get_param(self):
        header = self.header
        LOGGER.debug("In get_param, header: {}".format(header))
        if header is None:
            param_protobuf_obj = lr_model_param_pb2.LRModelParam()
            return param_protobuf_obj
        if self.need_one_vs_rest:
            one_vs_rest_class = list(map(str, self.one_vs_rest_obj.classes))
        else:
            one_vs_rest_class = None

        weight_dict = {}
        for idx, header_name in enumerate(header):
            if self.need_one_vs_rest:
                for class_idx, class_obj in enumerate(self.one_vs_rest_obj.models):
                    coef = class_obj.model_weights.coef_[idx]
                    class_type = one_vs_rest_class[class_idx]
                    class_and_header_name = "_".join(["class", str(class_type), header_name])
                    weight_dict[class_and_header_name] = coef
            else:
                coef_i = self.model_weights.coef_[idx]
                weight_dict[header_name] = coef_i

        if self.need_one_vs_rest:
            for class_idx, class_obj in enumerate(self.one_vs_rest_obj.models):
                intercept = class_obj.model_weights.intercept_
                class_type = one_vs_rest_class[class_idx]
                intercept_name = "_".join(["class", str(class_type), "intercept"])
                weight_dict[intercept_name] = intercept

            intercept_ = 0
        else:
            intercept_ = self.model_weights.intercept_

        param_protobuf_obj = lr_model_param_pb2.LRModelParam(iters=self.n_iter_,
                                                             loss_history=self.loss_history,
                                                             is_converged=self.is_converged,
                                                             weight=weight_dict,
                                                             intercept=intercept_,
                                                             header=header,
                                                             need_one_vs_rest=self.need_one_vs_rest,
                                                             one_vs_rest_classes=one_vs_rest_class
                                                             )
        json_result = json_format.MessageToJson(param_protobuf_obj)
        LOGGER.debug("json_result: {}".format(json_result))
        return param_protobuf_obj

    def _load_model(self, model_dict):
        result_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        fit_intercept = meta_obj.fit_intercept

        self.header = list(result_obj.header)
        # For hetero-lr arbiter predict function
        if self.header is None:
            return

        feature_shape = len(self.header)
        self.need_one_vs_rest = result_obj.need_one_vs_rest
        if self.need_one_vs_rest:
            self.one_vs_rest_classes = list(map(int, list(result_obj.one_vs_rest_classes)))
            weight_dict = dict(result_obj.weight)
            self.one_vs_rest_obj = OneVsRest(classifier=self, role=self.role, mode=self.mode,
                                             one_vs_rest_param=self._get_one_vs_rest_param())
            self.one_vs_rest_obj.classes = self.one_vs_rest_classes
            for class_type in self.one_vs_rest_obj.classes:
                classifier = copy.deepcopy(self)
                tmp_vars = np.zeros(feature_shape)
                for i, feature_name in enumerate(self.header):
                    feature_name = "_".join(["class", str(class_type), feature_name])
                    tmp_vars[i] = weight_dict.get(feature_name)
                intercept_name = "_".join(["class", str(class_type), "intercept"])
                tmp_intercept_ = weight_dict.get(intercept_name)
                if fit_intercept:
                    tmp_vars = np.append(tmp_vars, tmp_intercept_)
                classifier.model_weights = LogisticRegressionWeights(tmp_vars, fit_intercept=fit_intercept)
                self.one_vs_rest_obj.models.append(classifier)
        else:
            tmp_vars = np.zeros(feature_shape)
            weight_dict = dict(result_obj.weight)

            for idx, header_name in enumerate(self.header):
                tmp_vars[idx] = weight_dict.get(header_name)

            if fit_intercept:
                tmp_vars = np.append(tmp_vars, result_obj.intercept)
            self.model_weights = LogisticRegressionWeights(tmp_vars, fit_intercept=fit_intercept)

    def one_vs_rest_fit(self, train_data=None):
        self.need_one_vs_rest = True
        if self.role != consts.ARBITER:
            self.header = self.get_header(train_data)
        self.one_vs_rest_obj = OneVsRest(classifier=self, role=self.role, mode=self.mode,
                                         one_vs_rest_param=self._get_one_vs_rest_param())
        self.one_vs_rest_obj.fit(data_instances=train_data)

    def one_vs_rest_predict(self, validate_data):
        if not self.one_vs_rest_obj:
            LOGGER.warning("Not one_vs_rest fit before, return now")

        return self.one_vs_rest_obj.predict(data_instances=validate_data)

    def _get_one_vs_rest_param(self):
        return self.model_param.one_vs_rest_param

    def one_vs_rest_logic(self, stage, train_data=None, eval_data=None):
        LOGGER.info("Need one_vs_rest.")
        if stage == 'fit':
            self.one_vs_rest_fit(train_data)
            self.data_output = self.one_vs_rest_predict(train_data)
            if self.data_output:
                self.data_output = self.data_output.mapValues(lambda d: d + ["train"])

            if eval_data:
                eval_data_predict_res = self.one_vs_rest_predict(eval_data)
                if eval_data_predict_res:
                    predict_output_res = eval_data_predict_res.mapValues(lambda d: d + ["validation"])

                    if self.data_output:
                        self.data_output.union(predict_output_res)
                    else:
                        self.data_output = predict_output_res

            self.set_predict_data_schema(self.data_output, train_data.schema)

        elif stage == 'predict':
            self.data_output = self.one_vs_rest_predict(eval_data)
            if self.data_output:
                self.data_output = self.data_output.mapValues(lambda d: d + ["test"])
                self.set_predict_data_schema(self.data_output, eval_data.schema)
