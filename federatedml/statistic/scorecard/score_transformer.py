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


import numpy as np

from arch.api.utils import log_utils

from fate_flow.entity.metric import MetricMeta
from federatedml.model_base import ModelBase
from federatedml.param.scorecard_param import ScorecardParam
from federatedml.util.consts import FLOAT_ZERO

LOGGER = log_utils.getLogger()

class Scorecard(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = ScorecardParam()
        self.metric_name = "scorecard"
        self.metric_namespace = "train"
        self.metric_type = "SCORECARD"

    def _init_model(self, params):
        self.model_param = params
        self.method = params.method
        self.offset = params.offset
        self.factor = params.factor
        self.upper_limit_ratio = params.upper_limit_ratio
        self.lower_limit_value = params.lower_limit_value
        self.need_run = params.need_run

    @staticmethod
    def compute_credit_score(predict_result, offset, factor, upper_limit_value, lower_limit_value):
        predict_score = predict_result[3]["1"]
        odds = predict_result[3]["0"] / predict_score

        # deal with special predict score values
        if abs(predict_score - 0) < FLOAT_ZERO and predict_score >= 0:
            credit_score = upper_limit_value
        elif abs(predict_score - 1) < FLOAT_ZERO and predict_score > 0:
            credit_score = lower_limit_value
        elif predict_score > 1 or predict_score < 0:
            credit_score = -1
        else:
            credit_score = offset + factor / np.log(2) * np.log(odds)

        # credit score should be within range
        if credit_score > upper_limit_value:
            credit_score = upper_limit_value
        if credit_score < lower_limit_value:
            credit_score = lower_limit_value

        credit_score = round(credit_score, 2)

        return [predict_result[0], predict_result[1], predict_score, credit_score]

    def _callback(self):
        forumla = f"Score = {self.offset} + {self.factor} / ln2 * ln(Odds)"
        metas = {"scorecard_compute_formula": forumla}
        self.tracker.set_metric_meta(metric_namespace=self.metric_namespace,
                                     metric_name=self.metric_name,
                                     metric_meta=MetricMeta(name=self.metric_name,
                                                            metric_type=self.metric_type,
                                                            extra_metas=metas))
        LOGGER.info(f"Scorecard Computation Formula: {forumla}")

    def fit(self, prediction_result):
        LOGGER.info(f"Start Scorecard Transform, method: {self.method}")

        offset, factor = self.offset, self.factor
        upper_limit_value, lower_limit_value = self.upper_limit_ratio * offset, self.lower_limit_value
        score_result = prediction_result.mapValues(lambda v: Scorecard.compute_credit_score(v, offset, factor,
                                                                                            upper_limit_value,
                                                                                            lower_limit_value))
        schema = prediction_result.schema
        result_schema = {"header": ["label", "predict_result", "predict_score", "credit_score"],
                                   "sid_name": schema.get('sid_name')}
        score_result.schema = result_schema

        self._callback()
        LOGGER.info(f"Finish Scorecard Transform!")

        return score_result
