#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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

import logging
from typing import List

import pandas as pd

from fate.arch import Context
from ..abc.module import Module

logger = logging.getLogger(__name__)


class FeatureStatistics(Module):
    def __init__(self, metrics: List[str] = None, ddof=1, bias=True):
        self.metrics = metrics
        self.summary = StatisticsSummary(ddof, bias)

    def fit(self, ctx: Context, input_data, validate_data=None) -> None:
        self.summary.compute_metrics(input_data, self.metrics)

    def to_model(self):
        model = self.summary.to_model()
        model["metrics"] = self.metrics
        return model

    def restore(self, model):
        self.summary.restore(model)

    def from_model(cls, model) -> "FeatureStatistics":
        stat = FeatureStatistics(model["metrics"])
        stat.restore(model)
        return stat


class StatisticsSummary(Module):
    def __init__(self, ddof=1, bias=True):
        """if metrics is not None:
        if len(metrics) == 1 and metrics[0] == "describe":
            self.inner_metric_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        else:
            self.inner_metric_names = metrics"""
        self.ddof = ddof
        self.bias = bias
        self.inner_metric_names = []
        self.metrics_summary = None
        self._count = None
        self._nan_count = None
        self._mean = None
        self._describe = None

    def get_from_describe(self, data, metric):
        if self._describe is None:
            self._describe = data.describe(ddof=self.ddof, unbiased=~self.bias)
        return self._describe[metric]

    def compute_metrics(self, data, metrics):
        res = pd.DataFrame(columns=data.schema.columns)
        for metric in metrics:
            metric_val = None
            """if metric == "describe":
                res = data.describe(ddof=self.ddof, unbiased=~self.bias)
                self.metrics_summary = res
                self.inner_metric_names = list(res.index)
                return"""
            if metric in ["sum", "min", "max", "mean", "std", "var"]:
                metric_val = self.get_from_describe(data, metric)
            elif metric == "count":
                if self._count is None:
                    self._count = data.count()
                metric_val = self._count
            elif metric == "median":
                metric_val = data.median()
            elif metric == "coefficient_of_variation":
                metric_val = self.get_from_describe(data, "variation")
            elif metric == "missing_count":
                if self._nan_count is None:
                    self._nan_count = self.get_from_describe(data, "na_count")
                metric_val = self._nan_count
            elif metric == "missing_ratio":
                if self._nan_count is None:
                    self._nan_count = self.get_from_describe(data, "na_count")
                if self._count is None:
                    self._count = data.count()
                metric_val = self._nan_count / self._count
            elif metric == "skewness":
                metric_val = self.get_from_describe(data, "skew")
            elif metric == "kurtosis":
                metric_val = self.get_from_describe(data, "kurt")

            res.loc[metric] = metric_val

        has_nan = res.isnull().any()
        if has_nan.any():
            nan_cols = res.columns[has_nan].to_list()
            logger.warning(
                f"NaN value(s) found in statistics over columns: {nan_cols}; " f"this may lead to unexpected behavior."
            )
        self.metrics_summary = res
        self.inner_metric_names = list(res.index)

    def to_model(self):
        return {"inner_metric_names": self.inner_metric_names, "metrics_summary": self.metrics_summary.to_dict()}

    def restore(self, model):
        self.inner_metric_names = model["inner_metric_names"]
        self.metrics_summary = pd.DataFrame.from_dict(model["metrics_summary"])
