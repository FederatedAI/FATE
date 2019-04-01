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
from federatedml.util import LogisticParamChecker
from federatedml.util import consts
from federatedml.util import fate_operator

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
        self.data_shape = None

        self.gradient_operator = None
        self.initializer = Initializer()
        self.transfer_variable = None
        self.model_meta = LogisticRegressionModelMeta()
        self.loss_history = []
        self.is_converged = False
        self.header = None
        self.class_name = self.__class__.__name__

    def set_data_shape(self, data_shape):
        self.data_shape = data_shape

    def get_data_shape(self):
        return self.data_shape

    # def load_model(self, model_table, model_namespace):
    #
    #     LOGGER.debug("loading model, table: {}, namespace: {}".format(
    #         model_table, model_namespace))
    #     model = eggroll.table(model_table, model_namespace)
    #     model_local = model.collect()
    #     try:
    #         model_meta = model_local.__next__()[1]
    #     except StopIteration:
    #         LOGGER.warning("Cannot load model from name_space: {}, model_table: {}".format(
    #             model_namespace, model_table
    #         ))
    #         return
    #
    #     for meta_name, meta_value in model_meta.items():
    #         if not hasattr(self, meta_name):
    #             LOGGER.warning("Cannot find meta info {} in this model".format(meta_name))
    #             continue
    #         setattr(self, meta_name, meta_value)
    #
    # def save_model(self, model_table, model_namespace):
    #     meta_information = self.model_meta.__dict__
    #     save_dict = {}
    #     for meta_info in meta_information:
    #         if not hasattr(self, meta_info):
    #             LOGGER.warning("Cannot find meta info {} in this model".format(meta_info))
    #             continue
    #         save_dict[meta_info] = getattr(self, meta_info)
    #     LOGGER.debug("in save: {}".format(save_dict))
    #     meta_table = eggroll.parallelize([(1, save_dict)],
    #                                      include_key=True,
    #                                      name=model_table,
    #                                      namespace=model_namespace,
    #                                      error_if_exist=False,
    #                                      persistent=True
    #                                      )
    #     return meta_table

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

    def get_features_shape(self, data_instances):
        # LOGGER.debug("In get features shape method, data_instances count: {}".format(
        #     data_instances.count()
        # ))

        data_shape = self.get_data_shape()
        if data_shape is not None:
            return data_shape

        features = data_instances.collect()
        try:
            one_feature = features.__next__()
        except StopIteration:
            LOGGER.warning("Data instances is Empty")
            one_feature = None

        if one_feature is not None:
            return one_feature[1].features.shape[0]
        else:
            return None

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

    def save_model(self, name, namespace, job_id=None, model_name=None):
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
        LOGGER.debug("buffer_type is : {}".format(buffer_type))

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
