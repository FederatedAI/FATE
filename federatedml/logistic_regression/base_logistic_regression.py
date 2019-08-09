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
from federatedml.optim import DiffConverge, AbsConverge, Optimizer
from federatedml.optim import convergence
from federatedml.optim import Initializer
from federatedml.optim import L1Updater
from federatedml.optim import L2Updater
from federatedml.param.logistic_regression_param import LogisticParam
from federatedml.secureprotol import PaillierEncrypt, FakeEncrypt
from federatedml.statistic import data_overview
from federatedml.util import consts
from federatedml.util import fate_operator, abnormal_detection

LOGGER = log_utils.getLogger()


class BaseLogisticRegression(ModelBase):
    def __init__(self):
        super(BaseLogisticRegression, self).__init__()
        self.model_param = LogisticParam()
        # attribute:
        self.n_iter_ = 0
        self.coef_ = None
        self.intercept_ = 0
        self.classes_ = None
        self.feature_shape = None

        self.gradient_operator = None
        self.initializer = Initializer()
        self.transfer_variable = None
        self.loss_history = []
        self.is_converged = False
        self.header = None
        self.class_name = self.__class__.__name__
        self.model_name = 'LogisticRegression'
        self.model_param_name = 'LogisticRegressionParam'
        self.model_meta_name = 'LogisticRegressionMeta'
        self.role = ''
        self.mode = ''
        self.schema = {}

        # one_ve_rest parameter
        self.need_one_vs_rest = False
        self.one_vs_rest_classes = []
        self.one_vs_rest_obj = None

    def _init_model(self, params):
        self.model_param = params
        self.alpha = params.alpha
        self.init_param_obj = params.init_param
        self.fit_intercept = self.init_param_obj.fit_intercept
        self.learning_rate = params.learning_rate
        self.encrypted_mode_calculator_param = params.encrypted_mode_calculator_param
        self.encrypted_calculator = None

        if params.penalty == consts.L1_PENALTY:
            self.updater = L1Updater(self.alpha, self.learning_rate)
        elif params.penalty == consts.L2_PENALTY:
            self.updater = L2Updater(self.alpha, self.learning_rate)
        else:
            self.updater = None

        self.eps = params.eps
        self.batch_size = params.batch_size
        self.max_iter = params.max_iter
        self.learning_rate = params.learning_rate
        self.party_weight = params.party_weight
        self.penalty = params.penalty

        if params.encrypt_param.method == consts.PAILLIER:
            self.encrypt_operator = PaillierEncrypt()
        else:
            self.encrypt_operator = FakeEncrypt()

        if params.converge_func == 'diff':
            self.converge_func = DiffConverge(eps=self.eps)
        elif params.converge_func == 'weight_diff':
            self.converge_func = convergence.GradientConverge(eps=self.eps)
        else:
            self.converge_func = AbsConverge(eps=self.eps)
        self.re_encrypt_batches = params.re_encrypt_batches
        self.predict_param = params.predict_param
        self.optimizer = Optimizer(params.learning_rate, params.optimizer)
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

    def compute_wx(self, data_instances, coef_, intercept_=0):
        return data_instances.mapValues(lambda v: fate_operator.dot(v.features, coef_) + intercept_)

    def update_model(self, gradient):
        if self.fit_intercept:
            if self.updater is not None:
                self.coef_ = self.updater.update_coef(self.coef_, gradient[:-1])
            else:
                self.coef_ = self.coef_ - gradient[:-1]
            self.intercept_ -= gradient[-1]

        else:
            if self.updater is not None:
                self.coef_ = self.updater.update_coef(self.coef_, gradient)
            else:
                self.coef_ = self.coef_ - gradient

    def merge_model(self):
        w = self.coef_.copy()
        if self.fit_intercept:
            w = np.append(w, self.intercept_)
        return w

    def set_coef_(self, w):
        self.coef_ = []
        self.intercept_ = []
        if self.fit_intercept:
            self.coef_ = w[: -1]
            self.intercept_ = w[-1]
        else:
            self.coef_ = w
            self.intercept_ = 0

        LOGGER.debug("In set_coef_, coef: {}, intercept: {}, fit_intercept: {}".format(
            self.coef_, self.intercept_, self.fit_intercept
        ))

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
                                                          eps=self.eps,
                                                          alpha=self.alpha,
                                                          optimizer=self.model_param.optimizer,
                                                          party_weight=self.model_param.party_weight,
                                                          batch_size=self.batch_size,
                                                          learning_rate=self.learning_rate,
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
            self.one_vs_rest_classes = list(map(float, list(result_obj.one_vs_rest_classes)))
            weight = dict(result_obj.weight)
            self.one_vs_rest_obj = OneVsRest(classifier=self, role=self.role, mode=self.mode,
                                             one_vs_rest_param=self._get_one_vs_rest_param())
            self.one_vs_rest_obj.classes = self.one_vs_rest_classes
            for class_type in self.one_vs_rest_obj.classes:
                classifier = copy.deepcopy(self)
                classifier.coef_ = np.zeros(feature_shape)
                for i, feature_name in self.header:
                    feature_name = "_".join(["class", str(class_type), feature_name])
                    self.coef_[i] = weight.get(feature_name)
                # self.

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

    def update_local_model(self, fore_gradient, data_inst, coef, **training_info):
        """
        update local model that transforms features of raw input

        This 'update_local_model' function serves as a handler on updating local model that transforms features of raw
        input into more representative features. We typically adopt neural networks as the local model, which is
        typically updated/trained based on stochastic gradient descent algorithm. For concrete implementation, please
        refer to 'hetero_dnn_logistic_regression' folder.

        For this particular class (i.e., 'BaseLogisticRegression') that serves as a base class for neural-networks-based
        hetero-logistic-regression model, the 'update_local_model' function will do nothing. In other words, no updating
        performed on the local model since there is no one.

        Parameters:
        ___________
        :param fore_gradient: a table holding fore gradient
        :param data_inst: a table holding instances of raw input of guest side
        :param coef: coefficients of logistic regression model
        :param training_info: a dictionary holding training information
        """
        pass

    def transform(self, data_inst):
        """
        transform features of instances held by 'data_inst' table into more representative features

        This 'transform' function serves as a handler on transforming/extracting features from raw input 'data_inst' of
        guest. It returns a table that holds instances with transformed features. In theory, we can use any model to
        transform features. Particularly, we would adopt neural network models such as auto-encoder or CNN to perform
        the feature transformation task. For concrete implementation, please refer to 'hetero_dnn_logistic_regression'
        folder.

        For this particular class (i.e., 'BaseLogisticRegression') that serves as a base class for neural-networks-based
        hetero-logistic-regression model, the 'transform' function will do nothing but return whatever that has been
        passed to it. In other words, no feature transformation performed on the raw input of guest.

        Parameters:
        ___________
        :param data_inst: a table holding instances of raw input of guest side
        :return: a table holding instances with transformed features
        """
        return data_inst

    def cross_validation(self, data_instances):
        if not self.need_run:
            return data_instances
        kflod_obj = KFold()
        self.init_schema(data_instances)
        cv_param = self._get_cv_param()
        kflod_obj.run(cv_param, data_instances, self)
        LOGGER.debug("Finish kflod run")
        return data_instances

    def one_vs_rest(self, train_data=None, validate_data=None):
        self.need_one_vs_rest = True
        if self.role != consts.ARBITER:
            self.header = self.get_header(train_data)
        self.one_vs_rest_obj = OneVsRest(classifier=self, role=self.role, mode=self.mode,
                                    one_vs_rest_param=self._get_one_vs_rest_param())
        self.one_vs_rest_obj.fit(data_instances=train_data)
        train_data_predict_res = self.one_vs_rest_obj.predict(data_instances=train_data)

        output_res = None
        if train_data_predict_res:
            output_res = train_data_predict_res.mapValues(lambda d: d + ["train"])

        if validate_data:
            predict_data_predict_res = self.one_vs_rest_obj.predict(data_instances=validate_data)
            predict_output_res = predict_data_predict_res.mapValues(lambda d: d + ["predict"])
            if output_res:
                output_res.union(predict_output_res)
            else:
                output_res = predict_output_res

        return output_res


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