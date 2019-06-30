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


import numpy as np

from arch.api import eggroll
from federatedml.evaluation.evaluation import Evaluation


class TestEvaluationRun(object):
    def __init__(self):
        self.data_num = 5
        self.feature_num = 3
        final_result = []

        for i in range(self.data_num):
            tmp = [np.random.choice([0, 1]), np.random.random(), np.random.choice([0, 1]), "train"]
            tmp_pair = (str(i), tmp)
            final_result.append(tmp_pair)

        self.table = eggroll.parallelize(final_result,
                                         include_key=True,
                                         partition=10)

        self.model_name = 'Evaluation'
        self.args = {"data": {self.model_name: {"data": self.table}}}

    def _make_param_dict(self):
        component_param = {
            "EvaluateParam": {
                "metrics": ["auc", "precision", "ks", "roc", "gain", "lift", "recall", "accuracy", "explained_variance",
                            "mean_absolute_error",
                            "mean_squared_error", "mean_squared_log_error", "median_absolute_error", "r2_score",
                            "root_mean_squared_error"],
                "eval_type": "binary",
                "pos_label": 1
            }
        }

        return component_param

    def test_evaluation(self):
        eval_obj = Evaluation()
        component_param = self._make_param_dict()
        eval_obj.run(component_param, self.args)

    def tearDown(self):
        self.table.destroy()


if __name__ == '__main__':
    job_id = 'test_evaluation_2'
    eggroll.init(job_id)
    eval_obj = TestEvaluationRun()
    eval_obj.test_evaluation()
