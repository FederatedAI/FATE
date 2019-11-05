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

from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta
from federatedml.param.local_train_param import LocalTrainParam
from federatedml.model_base import ModelBase
from federatedml.statistic import data_overview
from federatedml.util import abnormal_detection

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

import numpy as np

LOGGER = log_utils.getLogger()
session.init("local_train")


class LocalTrain(ModelBase):
    def __init__(self):
        super(LocalTrain, self).__init__()
        self.model_param = LocalTrainParam()

    def _init_model(self, params):
        self.model_name = params.model_name
        self.need_cv = params.need_cv
        self.n_splits = params.n_splits
        self.shuffle = params.shuffle
        self.model_opts = params.model_opts
        self.random_state = params.random_state
        self.need_run = params.need_run


    def get_features_shape(self, data_instances):
        return data_overview.get_features_shape(data_instances)

    def get_model(self):
        model = LogisticRegression(**self.model_opts)
        return model

    def fit(self, data_instances):
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

        # get X, y arrays
        X_table = data_instances.mapValues(lambda v: v.features)
        y_table = data_instances.mapValues(lambda v: v.label)

        X = np.array(list(X_table.collect()))[:,1:]
        y = np.array(list(y_table.collect()))[:,1]
        # get model
        model = self.get_model()

        if self.need_cv:
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            return

        model_fit = model.fit(X, y)
        pred_label = data_instances.mapValues(lambda v: model_fit.predict(v.features))
        pred_prob = data_instances.mapValues(lambda v: model_fit.predict_proba(v.features))
        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
        predict_result = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1], {"0": x[1][0], "1": x[1][1]}])
        return predict_result






