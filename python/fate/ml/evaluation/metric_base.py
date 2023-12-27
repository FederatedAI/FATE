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

from typing import Dict
from transformers import EvalPrediction
from transformers.trainer_utils import PredictionOutput
import pandas as pd
import torch
import numpy as np
from typing import Union
import json
import logging

logger = logging.getLogger(__name__)


SINGLE_VALUE = "single_value"
TABLE_VALUE = "table_value"


class EvalResult(object):
    def __init__(self, metric_name: str, result: Union[int, float, pd.DataFrame]):
        self.metric_name = metric_name
        assert isinstance(self.metric_name, str), "metric_name must be a string."
        if isinstance(result, (int, float)):
            self.result = float(result)
            self.result_type = SINGLE_VALUE
        elif isinstance(result, pd.DataFrame):
            if len(result.shape) == 2:
                self.result = result
                self.result_type = TABLE_VALUE
            else:
                raise ValueError("DataFrame must be a 2D table.")
        else:
            raise TypeError("Invalid type for result. Expected int, float or DataFrame.")

    def __repr__(self) -> str:
        return self.result.__repr__()

    def to_dict(self):
        return {
            "metric": self.metric_name,
            # "result_type": self.result_type,
            "val": self.result.to_dict(orient="list") if self.result_type == TABLE_VALUE else self.result,
        }

    def to_json(self):
        if self.result_type == TABLE_VALUE:
            return self.result.to_json(orient="split")
        else:
            return json.dumps(self.to_dict())

    def get_raw_data(self):
        return self.result

    def __dict__(self):
        return self.to_dict()


class Metric(object):
    metric_name = None

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, predict, label, **kwargs) -> EvalResult:
        pass

    def to_np_format(self, data, flatten=True):
        if isinstance(data, list):
            ret = np.array(data)
        elif isinstance(data, torch.Tensor):
            ret = data.detach().cpu().numpy()
        elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            ret = np.array(data.values.tolist())
        else:
            ret = data

        if flatten:
            ret = ret.flatten()

        return ret.astype(np.float64)


class MetricEnsemble(object):
    def __init__(self, to_dict=True) -> None:
        self._metrics = []
        self._metric_suffix = set()
        self._to_dict = to_dict

    def add_metric(self, metric: Metric):
        self._metrics.append(metric)
        return self

    def _parse_input(self, eval_rs):
        if isinstance(eval_rs, EvalPrediction):
            # parse hugging face format
            predict = eval_rs.predictions
            label = eval_rs.label_ids
            input_ = eval_rs.inputs

        elif isinstance(eval_rs, PredictionOutput):
            predict = eval_rs.predictions
            label = eval_rs.label_ids
            input_ = None

        elif isinstance(eval_rs, tuple) and len(eval_rs) == 2:
            # conventional format
            predict, label = eval_rs
            input_ = None
        else:
            raise ValueError(
                "Unknown eval_rs format: {}. Expected input formats are either "
                "an instance of EvalPrediction or a 2-tuple (predict, label).".format(type(eval_rs))
            )

        return predict, label, input_

    def __call__(self, eval_rs=None, predict=None, label=None, **kwargs):
        metric_result = []

        if eval_rs is not None:
            predict, label, input_ = self._parse_input(eval_rs)

        for metric in self._metrics:
            rs = metric(predict, label)
            if isinstance(rs, tuple):
                new_rs = [r.to_dict() for r in rs]
                rs = new_rs
            elif isinstance(rs, EvalResult):
                rs = rs.to_dict()
            else:
                raise ValueError("cannot parse metric result: {}".format(rs))
            metric_result.append(rs)
        return metric_result

    def fit(self, eval_rs=None, predict=None, label=None, **kwargs):
        return self.__call__(eval_rs, predict, label, **kwargs)
