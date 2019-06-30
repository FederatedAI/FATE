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
from arch.api.utils import log_utils
from federatedml.util import consts
from federatedml.param.base_param import BaseParam

LOGGER = log_utils.getLogger()


class EvaluateParam(BaseParam):
    """
    Define the evaluation method of binary/multiple classification and regression

    Parameters
    ----------
    metrics: A list of evaluate index. Support 'auc', 'ks', 'lift', 'precision' ,'recall' and 'accuracy', 'explain_variance',
            'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error','median_absolute_error','r2_score','root_mean_squared_error'.
            For example, metrics can be set as ['auc', 'precision', 'recall'], then the results of these indexes will be output.

    eval_type: string, support 'binary' for HomoLR, HeteroLR and Secureboosting. support 'regression' for Secureboosting. 'multi' is not support these version

    pos_label: specify positive label type, can be int, float and str, this depend on the data's label, this parameter effective only for 'binary'

    thresholds: A list of threshold. Specify the threshold use to separate positive and negative class. for example [0.1, 0.3,0.5], this parameter effective only for 'binary'
    """

    def __init__(self, metrics=None, eval_type="binary", pos_label=1, thresholds=None):
        if metrics is None:
            metrics = []
        self.metrics = metrics
        self.eval_type = eval_type
        self.pos_label = pos_label
        self.thresholds = thresholds

    def check(self):
        if type(self.metrics).__name__ != "list":
            raise ValueError("evaluate param's metrics {} not supported, should be list".format(evaluate_param.metrics))
        else:
            descr = "evaluate param's metrics"
            for idx, metric in enumerate(self.metrics):
                self.metrics[idx] = self.check_and_change_lower(metric,
                                                                [consts.AUC, consts.KS, consts.LIFT,
                                                                 consts.PRECISION, consts.RECALL, consts.ACCURACY,
                                                                 consts.EXPLAINED_VARIANCE,
                                                                 consts.MEAN_ABSOLUTE_ERROR,
                                                                 consts.MEAN_SQUARED_ERROR,
                                                                 consts.MEAN_SQUARED_LOG_ERROR,
                                                                 consts.MEDIAN_ABSOLUTE_ERROR,
                                                                 consts.R2_SCORE, consts.ROOT_MEAN_SQUARED_ERROR,
                                                                 consts.ROC,
                                                                 consts.GAIN],
                                                                descr)
        descr = "evaluate param's "

        self.eval_type = self.check_and_change_lower(self.eval_type,
                                                       [consts.BINARY, consts.MULTY, consts.REGRESSION],
                                                       descr)

        if type(self.pos_label).__name__ not in ["str", "float", "int"]:
            raise ValueError(
                "evaluate param's pos_label {} not supported, should be str or float or int type".format(
                    self.pos_label))

        if self.thresholds is not None:
            if type(self.thresholds).__name__ != "list":
                raise ValueError(
                    "evaluate param's thresholds {} not supported, should be list".format(self.thresholds))
            else:
                for threshold in self.thresholds:
                    if type(threshold).__name__ not in ["float", "int"]:
                        raise ValueError(
                            "threshold {} in evaluate param's thresholds not supported, should be positive integer".format(
                                threshold))

        LOGGER.debug("Finish evaluation parameter check!")
        return True
