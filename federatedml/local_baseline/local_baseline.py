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

        # one_ve_rest parameter
        self.need_one_vs_rest = None
        self.one_vs_rest_classes = []
        self.one_vs_rest_obj = None

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
        model = self.model_fit
        n_iter = model.n_iter_[0]
        is_converged = n_iter < model.max_iter

        coef = model.coef_[0]
        LOGGER.debug(f"model coef len {coef.shape[0]}, value: {coef}")
        weight_dict = dict(zip(self.header, list(coef)))
        LOGGER.debug(f"model weight dict {weight_dict}")
        # intercept is in array format if fit_intercept
        intercept = model.intercept_[0] if model.fit_intercept else model.intercept_

        result = {'iters': n_iter,
                  'is_converged': is_converged,
                  'weight': weight_dict,
                  'intercept': intercept,
                  'header': self.header
                  }
        return result

    def _get_model_param_ovr(self):
        model = self.model_fit
        n_iter = model.n_iter_[0]
        is_converged = n_iter < model.max_iter
        classes = model.classes_
        coef_all = model.coef_
        intercept_all = model.intercept_
        ovr_pb_objs = []
        ovr_pb_classes = []

        for i, label in enumerate(classes):
            coef = coef_all[i,]
            weight_dict = dict(zip(self.header, list(coef)))
            intercept = intercept_all[i] if model.fit_intercept else intercept_all
            result = {'iters': n_iter,
                      'is_converged': is_converged,
                      'weight': weight_dict,
                      'intercept': intercept,
                      'header': self.header
                      }
            param_protobuf_obj = lr_model_param_pb2.SingleModel(**result)
            ovr_pb_objs.append(param_protobuf_obj)
            ovr_pb_classes.append(str(label))

        one_vs_rest_result = {
            'completed_models': ovr_pb_objs,
            'one_vs_rest_classes': ovr_pb_classes
        }
        return one_vs_rest_result

    def _get_param(self):
        header = self.header
        LOGGER.debug("In get_param, header: {}".format(header))
        if header is None:
            param_protobuf_obj = lr_model_param_pb2.LRModelParam()
            return param_protobuf_obj
        if self.need_one_vs_rest:
            result = self._get_model_param_ovr()
            param_protobuf_obj = lr_model_param_pb2.OneVsRestResult(**result)

        else:
            result = self._get_model_param()
            param_protobuf_obj = lr_model_param_pb2.LRModelParam(**result)

        LOGGER.debug("in _get_param, result: {}".format(result))

        return param_protobuf_obj

    def _get_meta(self):
        model = self.model_fit
        result = {'penalty': model.penalty,
                  'tol': model.tol,
                  'fit_intercept': model.fit_intercept,
                  'optimizer': model.solver,
                  'need_one_vs_rest': self.need_one_vs_rest,
                  'max_iter': model.max_iter
                  }
        meta_protobuf_obj = lr_model_meta_pb2.LRModelMeta(**result)

        return meta_protobuf_obj

    def export_model(self):
        if not self.need_run:
            return
        meta_obj = self._get_meta()
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
        classes = [int(x) for x in model_fit.classes_]
        pred_label = data_instances.mapValues(lambda v: model_fit.predict(v.features[None,:])[0])
        pred_prob = data_instances.mapValues(lambda v: model_fit.predict_proba(v.features[None,:])[0])

        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
        predict_result = predict_result.join(pred_label, lambda x, y: [x[0], int(y), x[1][classes.index(y)],
                                                                       dict(zip(classes, list(x[1])))])
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
        y = np.array([v[1] for v in list(y_table.collect())])

        self.model_fit = model.fit(X, y)
        self.need_one_vs_rest = len(self.model_fit.classes_) > 2
