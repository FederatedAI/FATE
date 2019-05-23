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

from arch.api.model_manager import manager as model_manager
from arch.api.proto import lr_model_meta_pb2, lr_model_param_pb2
from arch.api.utils import log_utils
from federatedml.evaluation import Evaluation
from federatedml.logistic_regression.logistic_regression_modelmeta import LogisticRegressionModelMeta
from federatedml.optim import Initializer
from federatedml.optim import L1Updater
from federatedml.optim import L2Updater
from federatedml.param import LogisticParam
from federatedml.secureprotol import PaillierEncrypt, FakeEncrypt
from federatedml.statistic import data_overview
from federatedml.util import LogisticParamChecker
from federatedml.util import consts
from federatedml.util import fate_operator, abnormal_detection

LOGGER = log_utils.getLogger()


class BaseLogisticRegression(object):
    def __init__(self, logistic_params: LogisticParam):
        self.param = logistic_params
        # set params
        LogisticParamChecker.check_param(logistic_params)
        self.alpha = logistic_params.alpha
        self.init_param_obj = logistic_params.init_param
        self.fit_intercept = self.init_param_obj.fit_intercept
        self.learning_rate = logistic_params.learning_rate
        self.encrypted_mode_calculator_param = logistic_params.encrypted_mode_calculator_param
        self.encrypted_calculator = None

        if logistic_params.penalty == consts.L1_PENALTY:
            self.updater = L1Updater(self.alpha, self.learning_rate)
        elif logistic_params.penalty == consts.L2_PENALTY:
            self.updater = L2Updater(self.alpha, self.learning_rate)
        else:
            self.updater = None

        self.eps = logistic_params.eps
        self.batch_size = logistic_params.batch_size
        self.max_iter = logistic_params.max_iter

        if logistic_params.encrypt_param.method == consts.PAILLIER:
            self.encrypt_operator = PaillierEncrypt()
        else:
            self.encrypt_operator = FakeEncrypt()

        # attribute:
        self.n_iter_ = 0
        self.coef_ = None
        self.intercept_ = 0
        self.classes_ = None
        self.feature_shape = None

        self.gradient_operator = None
        self.initializer = Initializer()
        self.transfer_variable = None
        self.model_meta = LogisticRegressionModelMeta()
        self.loss_history = []
        self.is_converged = False
        self.header = None
        self.class_name = self.__class__.__name__

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

    def set_flowid(self, flowid=0):
        if self.transfer_variable is not None:
            self.transfer_variable.set_flowid(flowid)
            LOGGER.debug("set flowid:" + str(flowid))

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

                # LOGGER.debug("intercept:" + str(self.intercept_))
                # LOGGER.debug("coef:" + str(self.coef_))

    def merge_model(self):
        w = self.coef_.copy()
        if self.fit_intercept:
            w = np.append(w, self.intercept_)
        return w

    def set_coef_(self, w):
        if self.fit_intercept:
            self.coef_ = w[: -1]
            self.intercept_ = w[-1]
        else:
            self.coef_ = w
            self.intercept_ = 0

    def classified(self, prob_table, threshold):
        """
        convert a probability table into a predicted class table.
        """
        predict_table = prob_table.mapValues(lambda x: 1 if x > threshold else 0)
        return predict_table

    def fit(self, data_instance):
        pass

    def predict(self, data_instance, predict_param):
        pass

    def evaluate(self, labels, pred_prob, pred_labels, evaluate_param):
        predict_res = None
        if evaluate_param.classi_type == consts.BINARY:
            predict_res = pred_prob
        elif evaluate_param.classi_type == consts.MULTY:
            predict_res = pred_labels
        else:
            LOGGER.warning("unknown classification type, return None as evaluation results")

        eva = Evaluation(evaluate_param.classi_type)
        return eva.report(labels, predict_res, evaluate_param.metrics, evaluate_param.thresholds,
                          evaluate_param.pos_label)

    def _save_meta(self, name, namespace):
        meta_protobuf_obj = lr_model_meta_pb2.LRModelMeta(penalty=self.param.penalty,
                                                          eps=self.eps,
                                                          alpha=self.alpha,
                                                          optimizer=self.param.optimizer,
                                                          party_weight=self.param.party_weight,
                                                          batch_size=self.batch_size,
                                                          learning_rate=self.learning_rate,
                                                          max_iter=self.max_iter,
                                                          converge_func=self.param.converge_func,
                                                          re_encrypt_batches=self.param.re_encrypt_batches)
        buffer_type = "{}.meta".format(self.class_name)

        model_manager.save_model(buffer_type=buffer_type,
                                 proto_buffer=meta_protobuf_obj,
                                 name=name,
                                 namespace=namespace)
        return buffer_type

    def save_model(self, name, namespace):
        meta_buffer_type = self._save_meta(name, namespace)
        # In case arbiter has no header
        header = self.header

        weight_dict = {}
        for idx, header_name in enumerate(header):
            coef_i = self.coef_[idx]
            weight_dict[header_name] = coef_i

        param_protobuf_obj = lr_model_param_pb2.LRModelParam(iters=self.n_iter_,
                                                             loss_history=self.loss_history,
                                                             is_converged=self.is_converged,
                                                             weight=weight_dict,
                                                             intercept=self.intercept_,
                                                             header=header)

        buffer_type = "{}.param".format(self.class_name)

        model_manager.save_model(buffer_type=buffer_type,
                                 proto_buffer=param_protobuf_obj,
                                 name=name,
                                 namespace=namespace)
        return [(meta_buffer_type, buffer_type)]

    def load_model(self, name, namespace):

        result_obj = lr_model_param_pb2.LRModelParam()
        buffer_type = "{}.param".format(self.class_name)

        model_manager.read_model(buffer_type=buffer_type,
                                 proto_buffer=result_obj,
                                 name=name,
                                 namespace=namespace)

        self.header = list(result_obj.header)
        feature_shape = len(self.header)
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

    def show_meta(self):
        meta_dict = {
            'penalty': self.param.penalty,
            'eps': self.eps,
            'alpha': self.alpha,
            'optimizer': self.param.optimizer,
            'party_weight': self.param.party_weight,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'converge_func': self.param.converge_func,
            're_encrypt_batches': self.param.re_encrypt_batches
        }

        LOGGER.info("Showing meta information:")
        for k, v in meta_dict.items():
            LOGGER.info("{} is {}".format(k, v))

    def show_model(self):
        model_dict = {
            'iters': self.n_iter_,
            'loss_history': self.loss_history,
            'is_converged': self.is_converged,
            'weight': self.coef_,
            'intercept': self.intercept_,
            'header': self.header
        }
        LOGGER.info("Showing model information:")
        for k, v in model_dict.items():
            LOGGER.info("{} is {}".format(k, v))

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
