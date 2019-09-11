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

from federatedml.protobuf.generated import lr_model_param_pb2, lr_model_meta_pb2
from arch.api.utils import log_utils
from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta
from federatedml.logistic_regression.logistic_regression_variables import LogisticRegressionVariables
from federatedml.model_base import ModelBase
from federatedml.model_selection.KFold import KFold
from federatedml.one_vs_rest.one_vs_rest import OneVsRest
from federatedml.optim import Initializer
from federatedml.optim.convergence import converge_func_factory
from federatedml.optim.optimizer import optimizer_factory
from federatedml.param.logistic_regression_param import LogisticParam
from federatedml.secureprotol import PaillierEncrypt, FakeEncrypt
from federatedml.statistic import data_overview
from federatedml.util import abnormal_detection
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class BaseLogisticRegression(ModelBase):
    def __init__(self):
        super(BaseLogisticRegression, self).__init__()
        self.model_param = LogisticParam()
        # attribute:
        self.n_iter_ = 0
        self.classes_ = None
        self.feature_shape = None
        self.gradient_operator = None
        self.initializer = Initializer()
        self.transfer_variable = None
        self.loss_history = []
        self.is_converged = False
        self.header = None
        self.model_name = 'LogisticRegression'
        self.model_param_name = 'LogisticRegressionParam'
        self.model_meta_name = 'LogisticRegressionMeta'
        self.role = ''
        self.mode = ''
        self.schema = {}
        self.cipher_operator = None

        # one_ve_rest parameter
        self.need_one_vs_rest = False
        self.one_vs_rest_classes = []
        self.one_vs_rest_obj = None
        self.lr_variables = None

    def _init_model(self, params):
        self.model_param = params
        self.alpha = params.alpha
        self.init_param_obj = params.init_param
        self.fit_intercept = self.init_param_obj.fit_intercept
        self.encrypted_mode_calculator_param = params.encrypted_mode_calculator_param
        self.encrypted_calculator = None

        self.batch_size = params.batch_size
        self.max_iter = params.max_iter
        self.party_weight = params.party_weight
        self.optimizer = optimizer_factory(params)

        if params.encrypt_param.method == consts.PAILLIER:
            self.cipher_operator = PaillierEncrypt()
        else:
            self.cipher_operator = FakeEncrypt()
        self.converge_func = converge_func_factory(params)
        self.re_encrypt_batches = params.re_encrypt_batches

    def set_feature_shape(self, feature_shape):
        self.feature_shape = feature_shape

    def set_header(self, header):
        self.header = header

    def get_features_shape(self, data_instances):
        if self.feature_shape is not None:
            return self.feature_shape
        return data_overview.get_features_shape(data_instances)

    def get_header(self, data_instances):
        if self.header is not None:
            return self.header
        return data_instances.schema.get("header")

    def callback_loss(self, iter_num, loss):
        if not self.need_one_vs_rest:
            metric_meta = MetricMeta(name='train',
                                     metric_type="LOSS",
                                     extra_metas={
                                         "unit_name": "iters",
                                     })
            # metric_name = self.get_metric_name('loss')

            self.callback_meta(metric_name='loss', metric_namespace='train', metric_meta=metric_meta)
            self.callback_metric(metric_name='loss',
                                 metric_namespace='train',
                                 metric_data=[Metric(iter_num, loss)])

    def compute_wx(self, data_instances, coef_, intercept_=0):
        return data_instances.mapValues(lambda v: np.dot(v.features, coef_) + intercept_)

    def fit(self, data_instance):
        pass

    def _get_meta(self):
        meta_protobuf_obj = lr_model_meta_pb2.LRModelMeta(penalty=self.model_param.penalty,
                                                          eps=self.model_param.eps,
                                                          alpha=self.alpha,
                                                          optimizer=self.model_param.optimizer,
                                                          party_weight=self.model_param.party_weight,
                                                          batch_size=self.batch_size,
                                                          learning_rate=self.model_param.learning_rate,
                                                          max_iter=self.max_iter,
                                                          converge_func=self.model_param.converge_func,
                                                          re_encrypt_batches=self.re_encrypt_batches,
                                                          fit_intercept=self.fit_intercept,
                                                          need_one_vs_rest=self.need_one_vs_rest)
        return meta_protobuf_obj

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
                    coef = class_obj.lr_variables.coef_[idx]
                    class_type = one_vs_rest_class[class_idx]
                    class_and_header_name = "_".join(["class", str(class_type), header_name])
                    weight_dict[class_and_header_name] = coef
            else:
                coef_i = self.lr_variables.coef_[idx]
                weight_dict[header_name] = coef_i

        if self.need_one_vs_rest:
            for class_idx, class_obj in enumerate(self.one_vs_rest_obj.models):
                intercept = class_obj.lr_variables.intercept_
                class_type = one_vs_rest_class[class_idx]
                intercept_name = "_".join(["class", str(class_type), "intercept"])
                weight_dict[intercept_name] = intercept

            intercept_ = 0
        else:
            intercept_ = self.lr_variables.intercept_

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

    def export_model(self):
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result

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
                classifier.lr_variables = LogisticRegressionVariables(l=tmp_vars, fit_intercept=fit_intercept)
                self.one_vs_rest_obj.models.append(classifier)
        else:
            tmp_vars = np.zeros(feature_shape)
            weight_dict = dict(result_obj.weight)

            for idx, header_name in enumerate(self.header):
                tmp_vars[idx] = weight_dict.get(header_name)

            if fit_intercept:
                tmp_vars = np.append(tmp_vars, result_obj.intercept)
            self.lr_variables = LogisticRegressionVariables(l=tmp_vars, fit_intercept=fit_intercept)

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

    def cross_validation(self, data_instances):
        if not self.need_run:
            return data_instances
        kflod_obj = KFold()
        self.init_schema(data_instances)
        cv_param = self._get_cv_param()
        kflod_obj.run(cv_param, data_instances, self)
        LOGGER.debug("Finish kflod run")
        return data_instances

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

    def _get_cv_param(self):
        self.model_param.cv_param.role = self.role
        self.model_param.cv_param.mode = self.mode
        return self.model_param.cv_param

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

    def set_schema(self, data_instance, header=None):
        if header is None:
            self.schema["header"] = self.header
        else:
            self.schema["header"] = header
        data_instance.schema = self.schema
        return data_instance

    def init_schema(self, data_instance):
        if data_instance is None:
            return
        self.schema = data_instance.schema
        self.header = self.schema.get('header')
