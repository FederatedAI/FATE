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

from arch.api import eggroll
from arch.api.utils import log_utils
from federatedml.evaluation import Evaluation
from federatedml.logistic_regression.logistic_regression_modelmeta import LogisticRegressionModelMeta
from federatedml.optim import Initializer
from federatedml.optim import L1Updater
from federatedml.optim import L2Updater
from federatedml.param import LogisticParam
from federatedml.secureprotol import PaillierEncrypt, FakeEncrypt
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.util import LogisticParamChecker

LOGGER = log_utils.getLogger()


class BaseLogisticRegression(object):
    def __init__(self, logistic_params: LogisticParam):
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

    def set_data_shape(self, data_shape):
        self.data_shape = data_shape

    def get_data_shape(self):
        return self.data_shape

    def load_model(self, model_table, model_namespace):

        LOGGER.debug("loading model, table: {}, namespace: {}".format(
            model_table, model_namespace))
        model = eggroll.table(model_table, model_namespace)
        model_local = model.collect()
        try:
            model_meta = model_local.__next__()[1]
        except StopIteration:
            LOGGER.warning("Cannot load model from name_space: {}, model_table: {}".format(
                model_namespace, model_table
            ))
            return

        for meta_name, meta_value in model_meta.items():
            if not hasattr(self, meta_name):
                LOGGER.warning("Cannot find meta info {} in this model".format(meta_name))
                continue
            setattr(self, meta_name, meta_value)

    def save_model(self, model_table, model_namespace):
        meta_information = self.model_meta.__dict__
        save_dict = {}
        for meta_info in meta_information:
            if not hasattr(self, meta_info):
                LOGGER.warning("Cannot find meta info {} in this model".format(meta_info))
                continue
            save_dict[meta_info] = getattr(self, meta_info)
        LOGGER.debug("in save: {}".format(save_dict))
        meta_table = eggroll.parallelize([(1, save_dict)],
                                         include_key=True,
                                         name=model_table,
                                         namespace=model_namespace,
                                         error_if_exist=False,
                                         persistent=True
                                         )
        return meta_table

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
