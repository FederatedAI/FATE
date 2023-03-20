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

import json
import logging
from typing import List

import pandas as pd

from fate.interface import Context
from ..abc.module import Module

logger = logging.getLogger(__name__)


class FeatureStatistics(Module):
    def __init__(self, metrics: List[str] = None, bias=True):
        self.metrics = metrics
        self.summary = StatisticsSummary(bias)

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        self.summary.compute_metrics(train_data, self.metrics)

    def to_model(self):
        return {"metrics": self.metrics,
                "inner_metrics": self.summary.inner_metric_names,
                "summary": self.summary.to_model()}

    def restore(self, model):
        self.summary.restore(model)

    def from_model(cls, model) -> "FeatureStatistics":
        stat = FeatureStatistics(model["metrics"])
        stat.restore(model)
        return stat


class StatisticsSummary(Module):
    def __init__(self, bias=True):
        """if metrics is not None:
            if len(metrics) == 1 and metrics[0] == "describe":
                self.inner_metric_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
            else:
                self.inner_metric_names = metrics"""
        self.bias = bias
        self.inner_metric_names = []
        self.summary = None
        self._count = None
        self._nan_count = None
        self._mean = None

    def compute_metrics(self, data, metrics):
        res = pd.DataFrame(columns=data.schema.columns)
        for metric in metrics:
            metric_val = None
            if metric == "describe":
                res = data.describe()
                self.summary = res
                self.inner_metric_names = list(res.index)
                return
            if metric == "count":
                if self._count is None:
                    self._count = data.count()
                    metric_val = self._count
            elif metric == "sum":
                metric_val = data.sum()
            elif metric == "max":
                metric_val = data.max()
            elif metric == "min":
                metric_val = data.min()
            elif metric == "mean":
                if self._mean is None:
                    self._mean = data.mean()
                metric_val = self._mean
            elif metric == "median":
                metric_val = data.median()
            elif metric == "std":
                metric_val = data.std()
            elif metric == "var":
                metric_val = data.var()
            elif metric == "coefficient_of_variation":
                metric_val = data.coe()
            elif metric == "missing_count":
                if self._nan_count is None:
                    self._nan_count = data.nan_count()
                metric_val = self._nan_count
            elif metric == "missing_ratio":
                if self._nan_count is None:
                    self._nan_count = data.nan_count()
                if self._count is None:
                    self._count = data.count()
                metric_val = self._nan_count / self._count
            elif metric == "skewness":
                metric_val = self.compute_skewness(data)
            elif metric == "kurtosis":
                metric_val = self.compute_kurtosis(data)

            res.loc[metric] = metric_val

        self.summary = res
        self.inner_metric_names = list(res.index)

    def compute_skewness(self, data):
        """
            The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness, i.e.
        .. math::
            g_1=\frac{m_3}{m_2^{3/2}}
        where
        .. math::
            m_i=\frac{1}{N}\\sum_{n=1}^N(x[n]-\bar{x})^i
        If the bias is False, return the adjusted Fisher-Pearson standardized moment coefficient
        i.e.
        .. math::
        G_1=\frac{k_3}{k_2^{3/2}}=
            \frac{\\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}.
        """
        if self._mean is None:
            self._mean = data.mean()
        m3 = ((data - self._mean) ** 3).mean()

        m2 = data.var(unbiased=False)
        zero_mask = m2 >= 1e-6
        m2[~zero_mask] = 1
        skewness = zero_mask * m3 / m2 ** 1.5
        if not self.bias:
            if self._count is None:
                self._count = data.count()
            bias_factor = (self._count * (self._count - 1)) ** 0.5 / (self._count - 2)
            skewness = skewness * bias_factor
        skewness[~zero_mask] = 0.0
        return skewness

    def compute_kurtosis(self, data):
        """
            Return Fisher coefficient of kurtosis:
            .. math::
                g = \frac{m_4}{m_2^2} - 3
            If bias is False, the calculations are corrected for statistical bias.
        """
        if self._mean is None:
            self._mean = data.mean()

        m4 = ((data - self._mean) ** 4).mean()
        m2 = data.var(unbiased=False)
        zero_mask = m2 >= 1e-6
        m2[~zero_mask] = 1
        kurtosis = m4 / m2 ** 2

        if not self.bias:
            if self._count is None:
                self._count = data.count()

            bias_factor = 1 / (self._count - 2) / (self._count - 3)
            new_val = bias_factor * ((self._count ** 2 - 1) * kurtosis - 3 * ((self._count - 1) ** 2)) + 3
            kurtosis[zero_mask] = new_val
        kurtosis -= 3
        kurtosis[~zero_mask] = 0.0
        return kurtosis

    def to_model(self):
        return {"inner_metrics": self.inner_metric_names,
                "summary": self.summary.to_json()}

    def restore(self, model):
        self.inner_metric_names = model["inner_metrics"]
        self.summary = pd.DataFrame.from_dict(json.loads(model["summary"]))
