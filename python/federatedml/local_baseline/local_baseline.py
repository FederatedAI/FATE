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

import copy

import numpy as np
from sklearn.linear_model import LogisticRegression

from federatedml.model_base import ModelBase
from federatedml.param.local_baseline_param import LocalBaselineParam
from federatedml.protobuf.generated import lr_model_meta_pb2, lr_model_param_pb2
from federatedml.statistic import data_overview
from federatedml.util import LOGGER
from federatedml.util import abnormal_detection
from federatedml.util.io_check import assert_io_num_rows_equal


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
        self.predict_param = params.predict_param
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
        n_iter = int(model.n_iter_[0])
        is_converged = bool(n_iter < model.max_iter)

        coef = model.coef_[0]
        #LOGGER.debug(f"model coef len {coef.shape[0]}, value: {coef}")
        weight_dict = dict(zip(self.header, [float(i) for i in coef]))
        #LOGGER.debug(f"model weight dict {weight_dict}")
        # intercept is in array format if fit_intercept
        intercept = model.intercept_[0] if model.fit_intercept else model.intercept_

        result = {'iters': n_iter,
                  'is_converged': is_converged,
                  'weight': weight_dict,
                  'intercept': intercept,
                  'header': self.header,
                  'best_iteration': -1
                  }
        return result

    def _get_model_param_ovr(self):
        model = self.model_fit
        n_iter = int(model.n_iter_[0])
        is_converged = bool(n_iter < model.max_iter)
        classes = model.classes_
        coef_all = model.coef_
        intercept_all = model.intercept_
        ovr_pb_objs = []
        ovr_pb_classes = []

        for i, label in enumerate(classes):
            coef = coef_all[i, ]
            weight_dict = dict(zip(self.header, list(coef)))
            intercept = intercept_all[i] if model.fit_intercept else intercept_all
            result = {'iters': n_iter,
                      'is_converged': is_converged,
                      'weight': weight_dict,
                      'intercept': intercept,
                      'header': self.header,
                      'best_iteration': -1
                      }
            param_protobuf_obj = lr_model_param_pb2.SingleModel(**result)
            ovr_pb_objs.append(param_protobuf_obj)
            ovr_pb_classes.append(str(label))

        one_vs_rest_result = {
            'completed_models': ovr_pb_objs,
            'one_vs_rest_classes': ovr_pb_classes
        }
        param_result = {'one_vs_rest_result': one_vs_rest_result,
                        'need_one_vs_rest': True,
                        'header': self.header}
        return param_result

    def _get_param(self):
        header = self.header
        #LOGGER.debug("In get_param, header: {}".format(header))
        if header is None:
            param_protobuf_obj = lr_model_param_pb2.LRModelParam()
            return param_protobuf_obj
        if self.need_one_vs_rest:
            result = self._get_model_param_ovr()
            param_protobuf_obj = lr_model_param_pb2.LRModelParam(**result)

        else:
            result = self._get_model_param()
            param_protobuf_obj = lr_model_param_pb2.LRModelParam(**result)

        #LOGGER.debug("in _get_param, result: {}".format(result))

        return param_protobuf_obj

    def _get_meta(self):
        model = self.model_fit
        predict_param = lr_model_meta_pb2.PredictMeta(**{"threshold": self.predict_param.threshold})
        result = {'penalty': model.penalty,
                  'tol': model.tol,
                  'fit_intercept': model.fit_intercept,
                  'optimizer': model.solver,
                  'need_one_vs_rest': self.need_one_vs_rest,
                  'max_iter': model.max_iter,
                  'predict_param': predict_param
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

    def get_model_summary(self):
        header = self.header
        if header is None:
            return {}
        if not self.need_one_vs_rest:
            param = self._get_model_param()
            summary = {
                'coef': param['weight'],
                'intercept': param['intercept'],
                'is_converged': param['is_converged'],
                'iters': param['iters'],
                'one_vs_rest': False
            }
        else:
            model = self.model_fit
            n_iter = int(model.n_iter_[0])
            is_converged = bool(n_iter < model.max_iter)
            classes = model.classes_
            coef_all = model.coef_
            intercept_all = model.intercept_
            summary = {}

            for i, label in enumerate(classes):
                coef = coef_all[i, ]
                weight_dict = dict(zip(self.header, [float(i) for i in coef]))
                intercept = float(intercept_all[i]) if model.fit_intercept else float(intercept_all)
                single_summary = {
                    'coef': weight_dict,
                    'intercept': intercept,
                    'is_converged': is_converged,
                    'iters': n_iter
                }
                single_key = f"{label}"
                summary[single_key] = single_summary
                summary['one_vs_rest'] = True
        return summary

    @assert_io_num_rows_equal
    def _load_single_coef(self, result_obj):
        feature_shape = len(self.header)
        tmp_vars = np.zeros(feature_shape)
        weight_dict = dict(result_obj.weight)
        for idx, header_name in enumerate(self.header):
            tmp_vars[idx] = weight_dict.get(header_name)
        return tmp_vars

    def _load_single_model(self, result_obj):
        coef = self._load_single_coef(result_obj)
        self.model_fit.__setattr__('coef_', np.array([coef]))
        self.model_fit.__setattr__('intercept_', np.array([result_obj.intercept]))
        self.model_fit.__setattr__('classes_', np.array([0, 1]))
        self.model_fit.__setattr__('n_iter_', [result_obj.iters])
        return

    def _load_ovr_model(self, result_obj):
        one_vs_rest_result = result_obj.one_vs_rest_result
        classes = np.array([int(i) for i in one_vs_rest_result.one_vs_rest_classes])
        models = one_vs_rest_result.completed_models

        class_count, feature_shape = len(classes), len(self.header)
        coef_all = np.zeros((class_count, feature_shape))
        intercept_all = np.zeros(class_count)
        iters = -1

        for i, label in enumerate(classes):
            model = models[i]
            coef = self._load_single_coef(model)
            coef_all[i, ] = coef
            intercept_all[i] = model.intercept
            iters = model.iters

        self.model_fit.__setattr__('coef_', coef_all)
        self.model_fit.__setattr__('intercept_', intercept_all)
        self.model_fit.__setattr__('classes_', classes)
        self.model_fit.__setattr__('n_iter_', [iters])
        return

    def _load_model_meta(self, meta_obj):
        self.model_fit.__setattr__('penalty', meta_obj.penalty)
        self.model_fit.__setattr__('tol', meta_obj.tol)
        self.model_fit.__setattr__('fit_intercept', meta_obj.fit_intercept)
        self.model_fit.__setattr__('solver', meta_obj.optimizer)
        self.model_fit.__setattr__('max_iter', meta_obj.max_iter)

    def load_model(self, model_dict):
        result_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        self.model_fit = LogisticRegression()
        self._load_model_meta(meta_obj)
        self.header = list(result_obj.header)

        self.need_one_vs_rest = meta_obj.need_one_vs_rest
        LOGGER.debug("in _load_model need_one_vs_rest: {}".format(self.need_one_vs_rest))
        if self.need_one_vs_rest:
            self._load_ovr_model(result_obj)
        else:
            self._load_single_model(result_obj)
        return

    @assert_io_num_rows_equal
    def predict(self, data_instances):
        if not self.need_run:
            return
        model_fit = self.model_fit
        classes = [int(x) for x in model_fit.classes_]
        if self.need_one_vs_rest:
            pred_prob = data_instances.mapValues(lambda v: model_fit.predict_proba(v.features[None, :])[0])

        else:
            pred_prob = data_instances.mapValues(lambda v: model_fit.predict_proba(v.features[None, :])[0][1])
        predict_result = self.predict_score_to_output(data_instances=data_instances, predict_score=pred_prob,
                                                      classes=classes, threshold=self.predict_param.threshold)
        return predict_result

    def fit(self, data_instances, validate_data=None):
        if not self.need_run:
            return
        # check if empty table
        LOGGER.info("Enter Local Baseline fit")
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

        w = None
        if data_overview.with_weight(data_instances):
            LOGGER.info(f"Input Data with Weight. Weight will be used to fit model.")
            weight_table = data_instances.mapValues(lambda v: v.weight)
            w = np.array([v[1] for v in list(weight_table.collect())])

        self.model_fit = model.fit(X, y, w)
        self.need_one_vs_rest = len(self.model_fit.classes_) > 2
        self.set_summary(self.get_model_summary())
