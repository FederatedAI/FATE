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

from arch.api.proto import lr_model_meta_pb2, lr_model_param_pb2
from arch.api.utils import log_utils
from federatedml.model_base import ModelBase
from federatedml.model_selection.KFold import KFold
from federatedml.one_vs_rest.one_vs_rest import OneVsRest
from federatedml.optim.convergence import converge_func_factory
from federatedml.optim import convergence
from federatedml.optim import Initializer
from federatedml.optim.optimizer import optimizer_factory
from fate_flow.entity.metric import MetricMeta
from fate_flow.entity.metric import Metric

from federatedml.param.logistic_regression_param import LogisticParam
from federatedml.secureprotol import PaillierEncrypt, FakeEncrypt
from federatedml.statistic import data_overview
from federatedml.util import consts
from federatedml.util import abnormal_detection

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

        self.encrypt_params = params.encrypt_param
        self.encrypt_method = self.encrypt_params.method
        self.converge_func = converge_func_factory(params)

        self.re_encrypt_batches = params.re_encrypt_batches
        self.predict_param = params.predict_param
        self.key_length = params.encrypt_param.key_length

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

    def classified(self, prob_table, threshold):
        """
        convert a probability table into a predicted class table.
        """
        predict_table = prob_table.mapValues(lambda x: 1 if x > threshold else 0)
        return predict_table

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
                                                          re_encrypt_batches=self.re_encrypt_batches)
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
                    coef = class_obj.coef_[idx]
                    class_type = one_vs_rest_class[class_idx]
                    class_and_header_name = "_".join(["class", str(class_type), header_name])
                    weight_dict[class_and_header_name] = coef
            else:
                coef_i = self.coef_[idx]
                weight_dict[header_name] = coef_i

        if self.need_one_vs_rest:
            for class_idx, class_obj in enumerate(self.one_vs_rest_obj.models):
                intercept = class_obj.intercept_
                class_type = one_vs_rest_class[class_idx]
                intercept_name = "_".join(["class", str(class_type), "intercept"])
                weight_dict[intercept_name] = intercept

            self.intercept_ = 0

        param_protobuf_obj = lr_model_param_pb2.LRModelParam(iters=self.n_iter_,
                                                             loss_history=self.loss_history,
                                                             is_converged=self.is_converged,
                                                             weight=weight_dict,
                                                             intercept=self.intercept_,
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
                classifier.coef_ = np.zeros(feature_shape)
                for i, feature_name in enumerate(self.header):
                    feature_name = "_".join(["class", str(class_type), feature_name])
                    classifier.coef_[i] = weight_dict.get(feature_name)
                intercept_name =  "_".join(["class", str(class_type), "intercept"])
                classifier.intercept_ = weight_dict.get(intercept_name)
                self.one_vs_rest_obj.models.append(classifier)
        else:
            self.coef_ = np.zeros(feature_shape)
            weight_dict = dict(result_obj.weight)
            self.intercept_ = result_obj.intercept

            for idx, header_name in enumerate(self.header):
                self.coef_[idx] = weight_dict.get(header_name)

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