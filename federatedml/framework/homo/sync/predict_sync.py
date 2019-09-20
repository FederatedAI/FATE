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

from federatedml.logistic_regression.logistic_regression_weights import LogisticRegressionWeights
from federatedml.optim import activation
from federatedml.util import consts
from federatedml.transfer_variable.transfer_class.base_transfer_variable import Variable


def classify(predict_wx, threshold):
    """
    convert a probability table into a predicted class table.
    """
    pred_prob = predict_wx.mapValues(lambda x: activation.sigmoid(x))
    predict_table = pred_prob.mapValues(lambda x: 1 if x > threshold else 0)

    return pred_prob, predict_table


class Arbiter(object):
    # noinspection PyAttributeOutsideInit
    def _register_predict_sync(self, predict_wx_variable: Variable,
                               final_model_variable: Variable,
                               predict_result_variable: Variable):
        self._predict_wx_variable = predict_wx_variable
        self._final_model_variable = final_model_variable
        self._predict_result_variable = predict_result_variable

    def start_predict(self, host_ciphers, lr_weights, predict_threshold, suffix=tuple()):
        # Send encrypted model to hosts.
        for idx, cipher in host_ciphers.items():
            if cipher is None:
                continue
            encrypted_lr_weights = lr_weights.encrypted(cipher, inplace=False)
            self._final_model_variable.remote(obj=encrypted_lr_weights.for_remote(),
                                              role=consts.HOST,
                                              idx=idx,
                                              suffix=suffix)

        # Receive wx results
        for idx, cipher in host_ciphers.items():
            if cipher is None:
                continue
            encrypted_predict_wx = self._predict_wx_variable.get(idx=idx, suffix=suffix)
            predict_wx = cipher.distribute_decrypt(encrypted_predict_wx)
            pred_prob, predict_table = classify(predict_wx, predict_threshold)
            self._predict_result_variable.remote(predict_table,
                                                 role=consts.HOST,
                                                 idx=idx,
                                                 suffix=suffix)


class Host(object):
    # noinspection PyAttributeOutsideInit
    def _register_predict_sync(self, predict_wx_variable: Variable,
                               final_model_variable: Variable,
                               predict_result_variable: Variable):
        self._predict_wx_variable = predict_wx_variable
        self._final_model_variable = final_model_variable
        self._predict_result_variable = predict_result_variable

    def _register_func(self, compute_wx):
        self.compute_wx = compute_wx

    def start_predict(self, data_instances, lr_weights, predict_threshold,
                      use_encrypted, fit_intercept, suffix=tuple()):
        if use_encrypted:
            final_model = self._final_model_variable.get(idx=0, suffix=suffix)
            lr_weights = LogisticRegressionWeights(final_model.unboxed, fit_intercept)

        wx = self.compute_wx(data_instances, lr_weights.coef_, lr_weights.intercept_)
        if use_encrypted:
            self._predict_wx_variable.remote(wx, consts.ARBITER, 0, suffix)
            predict_result = self._predict_result_variable.get(idx=0, suffix=suffix)
            predict_result_table = predict_result.join(data_instances, lambda p, d: [d.label, None, p,
                                                                                     {"0": None, "1": None}])
        else:
            pred_prob, pred_label = classify(wx, predict_threshold)
            predict_result = data_instances.mapValues(lambda x: x.label)
            predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
            predict_result_table = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1],
                                                                                  {"1": x[1], "0": (1 - x[1])}])

        return predict_result_table


class Guest(object):
    # noinspection PyAttributeOutsideInit
    def _register_predict_sync(self, predict_wx_variable: Variable,
                               final_model_variable: Variable,
                               predict_result_variable: Variable):
        self._predict_wx_variable = predict_wx_variable
        self._final_model_variable = final_model_variable
        self._predict_result_variable = predict_result_variable

    def _register_func(self, compute_wx):
        self.compute_wx = compute_wx

    def start_predict(self, data_instances, lr_weights, predict_threshold):
        wx = self.compute_wx(data_instances, lr_weights.coef_, lr_weights.intercept_)
        pred_prob, pred_label = classify(wx, predict_threshold)

        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
        predict_result = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1], {"1": x[1], "0": (1 - x[1])}])
        return predict_result
