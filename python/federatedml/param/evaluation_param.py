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
#
from federatedml.util import consts, LOGGER
from federatedml.param.base_param import BaseParam


class EvaluateParam(BaseParam):
    """
    Define the evaluation method of binary/multiple classification and regression

    Parameters
    ----------
    eval_type : {'binary', 'regression', 'multi'}
        support 'binary' for HomoLR, HeteroLR and Secureboosting,
        support 'regression' for Secureboosting,
        'multi' is not support these version

    unfold_multi_result : bool
        unfold multi result and get several one-vs-rest binary classification results

    pos_label : int or float or str
        specify positive label type, depend on the data's label. this parameter effective only for 'binary'

    need_run: bool, default True
        Indicate if this module needed to be run
    """

    def __init__(self, eval_type="binary", pos_label=1, need_run=True, metrics=None,
                 run_clustering_arbiter_metric=False, unfold_multi_result=False):
        super().__init__()
        self.eval_type = eval_type
        self.pos_label = pos_label
        self.need_run = need_run
        self.metrics = metrics
        self.unfold_multi_result = unfold_multi_result
        self.run_clustering_arbiter_metric = run_clustering_arbiter_metric

        self.default_metrics = {
            consts.BINARY: consts.ALL_BINARY_METRICS,
            consts.MULTY: consts.ALL_MULTI_METRICS,
            consts.REGRESSION: consts.ALL_REGRESSION_METRICS,
            consts.CLUSTERING: consts.ALL_CLUSTER_METRICS
        }

        self.allowed_metrics = {
            consts.BINARY: consts.ALL_BINARY_METRICS,
            consts.MULTY: consts.ALL_MULTI_METRICS,
            consts.REGRESSION: consts.ALL_REGRESSION_METRICS,
            consts.CLUSTERING: consts.ALL_CLUSTER_METRICS
        }

    def _use_single_value_default_metrics(self):

        self.default_metrics = {
            consts.BINARY: consts.DEFAULT_BINARY_METRIC,
            consts.MULTY: consts.DEFAULT_MULTI_METRIC,
            consts.REGRESSION: consts.DEFAULT_REGRESSION_METRIC,
            consts.CLUSTERING: consts.DEFAULT_CLUSTER_METRIC
        }

    def _check_valid_metric(self, metrics_list):

        metric_list = consts.ALL_METRIC_NAME
        alias_name: dict = consts.ALIAS

        full_name_list = []

        metrics_list = [str.lower(i) for i in metrics_list]

        for metric in metrics_list:

            if metric in metric_list:
                if metric not in full_name_list:
                    full_name_list.append(metric)
                continue

            valid_flag = False
            for alias, full_name in alias_name.items():
                if metric in alias:
                    if full_name not in full_name_list:
                        full_name_list.append(full_name)
                    valid_flag = True
                    break

            if not valid_flag:
                raise ValueError('metric {} is not supported'.format(metric))

        allowed_metrics = self.allowed_metrics[self.eval_type]

        for m in full_name_list:
            if m not in allowed_metrics:
                raise ValueError('metric {} is not used for {} task'.format(m, self.eval_type))

        if consts.RECALL in full_name_list and consts.PRECISION not in full_name_list:
            full_name_list.append(consts.PRECISION)

        if consts.RECALL not in full_name_list and consts.PRECISION in full_name_list:
            full_name_list.append(consts.RECALL)

        return full_name_list

    def check(self):

        descr = "evaluate param's "
        self.eval_type = self.check_and_change_lower(self.eval_type,
                                                     [consts.BINARY, consts.MULTY, consts.REGRESSION,
                                                      consts.CLUSTERING],
                                                     descr)

        if type(self.pos_label).__name__ not in ["str", "float", "int"]:
            raise ValueError(
                "evaluate param's pos_label {} not supported, should be str or float or int type".format(
                    self.pos_label))

        if type(self.need_run).__name__ != "bool":
            raise ValueError(
                "evaluate param's need_run {} not supported, should be bool".format(
                    self.need_run))

        if self.metrics is None or len(self.metrics) == 0:
            self.metrics = self.default_metrics[self.eval_type]
            LOGGER.warning('use default metric {} for eval type {}'.format(self.metrics, self.eval_type))

        self.check_boolean(self.unfold_multi_result, 'multi_result_unfold')

        self.metrics = self._check_valid_metric(self.metrics)

        LOGGER.info("Finish evaluation parameter check!")

        return True

    def check_single_value_default_metric(self):
        self._use_single_value_default_metrics()

        # in validation strategy, psi f1-score and confusion-mat pr-quantile are not supported in cur version
        if self.metrics is None or len(self.metrics) == 0:
            self.metrics = self.default_metrics[self.eval_type]
            LOGGER.warning('use default metric {} for eval type {}'.format(self.metrics, self.eval_type))

        ban_metric = [consts.PSI, consts.F1_SCORE, consts.CONFUSION_MAT, consts.QUANTILE_PR]
        for metric in self.metrics:
            if metric in ban_metric:
                self.metrics.remove(metric)
        self.check()
