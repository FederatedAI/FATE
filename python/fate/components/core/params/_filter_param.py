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

from typing import Union, List

import pydantic

from ._fields import string_choice, Parameter, conint, confloat
from ._metrics import statistic_metrics_param, legal_percentile


class StandardFilterParam(pydantic.BaseModel, Parameter):
    metrics: List[str]

    filter_type: List[string_choice({"threshold", "top_k", "top_percentile"})] = ["threshold"]
    threshold: List[Union[confloat(ge=0.0, le=1.0), conint(ge=1)]] = [1.0]
    take_high: List[bool] = [True]

    @pydantic.validator("metrics", "filter_type", "threshold", "take_high", pre=True, allow_reuse=True)
    def to_list(cls, v):
        return v if isinstance(v, list) else [v]

    @pydantic.root_validator(pre=False)
    def check_filter_param_length(cls, values):
        max_length = max([len(x) for k, x in values.items()])
        for k, v in values.items():
            if len(v) == 1:
                v *= max_length
            assert len(v) == max_length, (
                f"Length of {k}: {v} does not match "
                f"max length {max_length} of (metrics, filter_type, threshold, take_high)."
            )
        return values


class FederatedStandardFilterParam(StandardFilterParam, Parameter):
    host_filter_type: List[string_choice({"threshold", "top_k", "top_percentile"})] = ["threshold"]
    host_threshold: List[Union[confloat(ge=0.0, le=1.0), conint(ge=1)]] = [1.0]
    host_take_high: List[bool] = [True]

    select_federated: bool = True

    @pydantic.validator("host_filter_type", "host_threshold", "host_take_high", pre=True, allow_reuse=True)
    def to_list(cls, v):
        return v if isinstance(v, list) else [v]

    @pydantic.root_validator(pre=False)
    def check_filter_param_length(cls, values):
        select_values = {k: v for k, v in values.items() if k != "select_federated"}
        max_length = max([len(x) for k, x in select_values.items()])
        for k, v in select_values.items():
            if len(v) == 1:
                v *= max_length
            assert len(v) == max_length, (
                f"Length of {k}: {v} does not match "
                f"max length {max_length} of (metrics, filter_type, threshold, take_high)."
            )
        return values


class IVFilterParam(FederatedStandardFilterParam, Parameter):
    metrics: List[string_choice({"iv"})] = ["iv"]


class StatisticFilterParam(StandardFilterParam, Parameter):
    metrics: List[Union[statistic_metrics_param(), legal_percentile()]] = ["mean"]


class ManualFilterParam(pydantic.BaseModel, Parameter):
    keep_col: List[str] = []
    filter_out_col: List[str] = []

    @pydantic.root_validator(pre=False)
    def no_intersection(cls, values):
        filter_out_col = values.get("filter_out_col", [])
        keep_col = values.get("keep_col", [])
        intersection = set(filter_out_col).intersection(set(keep_col))
        if intersection:
            raise ValueError(f"`keep_col` and `filter_out_col` share common elements: {intersection}")
        return values


def iv_filter_param():
    namespace = {}
    return type("IVFilterParam", (IVFilterParam,), namespace)


def statistic_filter_param():
    namespace = {}
    return type("StatisticFilterParam", (StatisticFilterParam,), namespace)


def manual_filter_param():
    namespace = {}
    return type("ManualFilterParam", (ManualFilterParam,), namespace)
