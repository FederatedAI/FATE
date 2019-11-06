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

from federatedml.evaluation import Evaluation
from federatedml.param.local_baseline_model_param import LocalBaselineModelParam
from federatedml.model_base import ModelBase
from federatedml.util import abnormal_detection

from sklearn.linear_model import LogisticRegression

import numpy as np

LOGGER = log_utils.getLogger()
session.init("local_baseline_model")


class LocalBaselineModel(ModelBase):
    def __init__(self):
        super(LocalBaselineModel, self).__init__()
        self.model_param = LocalBaselineModelParam()

    def _init_model(self, params):
        self.model_name = params.model_name
        self.model_opts = params.model_opts
        self.need_run = params.need_run

    def get_model(self):
        # extend in future with more model types
        model = LogisticRegression(**self.model_opts)
        return model

    def run_predict(self, model_fit, data_instances):
        pred_label = data_instances.mapValues(lambda v: model_fit.predict(v.features))
        pred_prob = data_instances.mapValues(lambda v: model_fit.predict_proba(v.features))
        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
        predict_result = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1], {"0": x[1][0], "1": x[1][1]}])
        return predict_result

    def fit(self, data_instances):
        # check if empty table
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
        # get model
        model = self.get_model()

        X_table = data_instances.mapValues(lambda v: v.features)
        y_table = data_instances.mapValues(lambda v: v.label)

        X = np.array([v[1] for v in list(X_table.collect())])
        y = np.array(list(y_table.collect()))[:, 1]
        model_fit = model.fit(X, y)
        results = self.run_predict(model_fit, data_instances)
        return results
