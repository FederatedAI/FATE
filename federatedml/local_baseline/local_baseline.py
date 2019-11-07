#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from arch.api.utils import log_utils
from arch.api import session

import copy

from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.model_base import ModelBase
from federatedml.param.local_baseline_param import LocalBaselineParam
from federatedml.util import abnormal_detection
from federatedml.statistic import data_overview

from sklearn.linear_model import LogisticRegression

import numpy as np

LOGGER = log_utils.getLogger()
session.init("baseline")


class LocalBaseline(ModelBase):
    def __init__(self):
        super(LocalBaseline, self).__init__()
        self.model_param = LocalBaselineParam()
        self.model_name = "LocalBaseline"

    def _init_model(self, params):
        self.model_name = params.model_name
        self.model_opts = params.model_opts
        self.need_run = params.need_run
        self.model = None
        self.model_fit = None
        self.header = None
        self.model_weights = None
        self.fit_intercept = True

    def get_model(self):
        # extend in future with more model types
        self.fit_intercept = self.model_opts.get("fit_intercept", True)
        model = LogisticRegression(**self.model_opts)
        self.model = copy.deepcopy(model)
        return model

    def get_model_param(self, coef, intercept):
        weight_dict = {}
        for idx, header_name in enumerate(self.header):
            coef_i = coef[idx]
            weight_dict[header_name] = coef_i

        result = {'iters': self.n_iter_,
                  'is_converged': self.is_converged,
                  'weight': weight_dict,
                  'intercept': intercept,
                  'header': self.header
                  }
        return result

    def predict(self, data_instances):
        model_fit = self.model_fit
        pred_label = data_instances.mapValues(lambda v: model_fit.predict(v.features[None,:])[0])
        pred_prob = data_instances.mapValues(lambda v: model_fit.predict_proba(v.features[None,:])[0])
        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
        predict_result = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1][1], {"0": x[1][0], "1": x[1][1]}])
        LOGGER.debug("pred_result is {}".format(list(predict_result.collect())))
        return predict_result

    def fit(self, data_instances, validate_data=None):
        # check if empty table
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
        # get model
        model = self.get_model()
        # get header
        self.header = data_overview.get_header(data_instances)

        X_table = data_instances.mapValues(lambda v: v.features)
        y_table = data_instances.mapValues(lambda v: v.label)

        X = np.array([v[1] for v in list(X_table.collect())])
        y = np.array(list(y_table.collect()))[:, 1]

        self.model_fit = model.fit(X, y)
        #@TODO: get model output: convergence, iterations, coefficients & intercetp
        coef = model.coef_
        intercept  = model.intercept_
        #w = np.hstack((coef, intercept[None,:]))
        #self.model_weights = LinearModelWeights(w, fit_intercept=self.fit_intercept)
        model_param = self.get_model_param(coef, intercept)


