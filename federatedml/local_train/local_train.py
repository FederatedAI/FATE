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
from federatedml.evaluation import Evaluation
from federatedml.param.local_train_param import LocalTrainParam
from federatedml.param.evaluation_param import EvaluateParam
from federatedml.model_base import ModelBase
from federatedml.model_selection.k_fold import KFold
from federatedml.statistic import data_overview
from federatedml.util import abnormal_detection

#from sklearn.model_selection import KFold
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
        self.random_seed = params.random_seed
        self.need_run = params.need_run

    def get_metrics_param(self):
        # extend in future with more model types
        param =  EvaluateParam(eval_type="binary",
                             pos_label=1)
        return param

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

    def run_evaluate(self, eval_data, fold_name):
        eval_obj = Evaluation()
        eval_param = self.get_metrics_param()
        eval_obj._init_model(eval_param)
        eval_data = {fold_name: eval_data}
        eval_obj.fit(eval_data)
        eval_obj.save_data()

    def fit(self, data_instances):
        # check if empty table
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
        # get model
        model = self.get_model()

        if not self.need_cv:
            X_table = data_instances.mapValues(lambda v: v.features)
            y_table = data_instances.mapValues(lambda v: v.label)

            X = np.array([v[1] for v in list(X_table.collect())])
            y = np.array(list(y_table.collect()))[:, 1]
            model_fit = model.fit(X, y)
            results = self.run_predict(model_fit, data_instances)
            return results

        kf = KFold()
        kf.n_splits = self.n_splits
        kf.shuffle = self.shuffle
        kf.random_seed = self.random_seed
        generator = kf.split(data_instances)

        fold_num = 0
        for data_train, data_test in generator:
            X_table = data_train.mapValues(lambda v: v.features)
            y_table = data_train.mapValues(lambda v: v.label)

            X_train = np.array([v[1] for v in list(X_table.collect())])
            y_train = np.array(list(y_table.collect()))[:, 1]

            model_fit = model.fit(X_train, y_train)
            predict_train = self.run_predict(model_fit, data_train)
            fold_name = "_".join(['train', 'fold', str(fold_num)])
            pred_res = predict_train.mapValues(lambda value: value + ['train'])
            self.run_evaluate(pred_res, fold_name)

            predict_test = self.run_predict(model_fit, data_test)
            fold_name = "_".join(['test', 'fold', str(fold_num)])
            pred_res = predict_test.mapValues(lambda value: value + ['test'])
            self.run_evaluate(pred_res, fold_name)

            fold_num += 1
