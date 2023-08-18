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

import re
from typing import Type

from ._fields import StringChoice, Parameter


class Metrics(StringChoice):
    choice = {}


class StatisticMetrics(StringChoice):
    choice = {}


def statistic_metrics_param(
    count=True,
    sum=True,
    min=True,
    max=True,
    mean=True,
    median=True,
    std=True,
    var=True,
    coe=True,
    missing_count=True,
    missing_ratio=True,
    skewness=True,
    kurtosis=True
) -> Type[str]:
    choice = {
        "count": count,
        "sum": sum,
        "max": max,
        "min": min,
        "mean": mean,
        "median": median,
        "std": std,
        "var": var,
        "coefficient_of_variation": coe,
        "missing_count": missing_count,
        "missing_ratio": missing_ratio,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }
    namespace = dict(
        choice={k for k, v in choice.items() if v},
    )
    return type("StatisticMetrics", (StatisticMetrics,), namespace)


def metrics_param(auc=True, ks=True, accuracy=True, mse=True) -> Type[str]:
    choice = {"auc": auc, "ks": ks, "accuracy": accuracy, "mse": mse}
    namespace = dict(
        choice={k for k, v in choice.items() if v},
    )
    return type("Metrics", (Metrics,), namespace)


class LegalPercentile(str, Parameter):
    legal_percentile = r"^(100|\d{1,2})%$"

    @classmethod
    def __get_validators__(cls):
        yield cls.percentile_validator

    @classmethod
    def percentile_validator(cls, v):
        if re.match(cls.legal_percentile, v):
            return v
        raise ValueError(f"provided `{v}` not in legal percentile format")


def legal_percentile() -> Type[str]:
    namespace = dict(
        legal_percentile=LegalPercentile.legal_percentile,
    )
    return type("LegalPercentile", (LegalPercentile,), namespace)
