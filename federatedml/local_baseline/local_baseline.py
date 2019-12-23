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

from federatedml.model_base import ModelBase
from federatedml.param.local_baseline_param import LocalBaselineParam
from federatedml.protobuf.generated import lr_model_meta_pb2
from federatedml.protobuf.generated import lr_model_param_pb2
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
        self.metric_type = ""
        self.model_param_name = "LocalBaselineParam"
        self.model_meta_name = "LocalBaselineMeta"

    def _init_model(self, params):
        self.model_name = params.model_name
        self.model_opts = params.model_opts
        self.model = None
        self.model_fit = None
        self.header = None
        self.model_weights = None

    def get_model(self):
        # extend in future with more model types
        model = LogisticRegression(**self.model_opts)
        self.model = copy.deepcopy(model)
        return model

    def _get_model_param(self):
        weight_dict = {}
        n_iter = self.model_fit.n_iter_[0]
        is_converged = n_iter < self.model_fit.max_iter
        coef = self.model_fit.coef_[0]
        intercept = self.model_fit.intercept_[0]
        for idx, header_name in enumerate(self.header):
            coef_i = coef[idx]
            weight_dict[header_name] = coef_i

        result = {'iters': n_iter,
                  'is_converged': is_converged,
                  'weight': weight_dict,
                  'intercept': intercept,
                  'header': self.header,
                  }
        return result

    def _get_param(self):
        header = self.header
        if header is None:
            param_protobuf_obj = lr_model_param_pb2.LRModelParam()
            return param_protobuf_obj
        result = self._get_model_param()
        param_protobuf_obj = lr_model_param_pb2.LRModelParam(**result)
        return param_protobuf_obj

    def export_model(self):
        meta_obj = lr_model_meta_pb2.LRModelMeta()
        param_obj = self._get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result

    def predict(self, data_instances):
        if not self.need_run:
            return
        model_fit = self.model_fit
        pred_label = data_instances.mapValues(lambda v: model_fit.predict(v.features[None,:])[0])
        pred_prob = data_instances.mapValues(lambda v: model_fit.predict_proba(v.features[None,:])[0])
        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
        predict_result = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1][1], {"0": x[1][0], "1": x[1][1]}])
        return predict_result

    def fit(self, data_instances, validate_data=None):
        if not self.need_run:
            return
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
