from typing import Union, List

import pydantic

from ._fields import string_choice
from ._metrics import statistic_metrics_param


class StandardFilterParam(pydantic.BaseModel):
    metrics: List[str]

    filter_type: List[string_choice({'threshold', 'top_k', 'top_percentile'})] = ['threshold']
    threshold: List[Union[int, float]] = [1.0]
    take_high: List[bool] = [True]

    @pydantic.validator('metrics', 'filter_type', 'threshold', 'take_high', pre=True, allow_reuse=True)
    def to_list(cls, v):
        return v if isinstance(v, list) else [v]

    @pydantic.root_validator(pre=False)
    def check_filter_param_length(cls, values):
        max_length = max([len(x) for k, x in values.items()])
        for k, v in values.items():
            if len(v) == 1:
                v *= max_length
            assert len(v) == max_length, f"Length of {k}: {v} does not match " \
                                         f"max length {max_length} of (metrics, filter_type, threshold, take_high)."
        return values


class FederatedStandardFilterParam(StandardFilterParam):
    host_filter_type: List[string_choice({'threshold', 'top_k', 'top_percentile'})] = ['threshold']
    host_threshold: List[Union[int, float]] = [1.0]
    host_take_high: List[bool] = [True]

    select_federated: bool = True

    @pydantic.validator('host_filter_type', 'host_threshold', 'host_take_high', pre=True, allow_reuse=True)
    def to_list(cls, v):
        return v if isinstance(v, list) else [v]

    @pydantic.root_validator(pre=False)
    def check_filter_param_length(cls, values):
        select_values = {k: v for k, v in values.items() if k != 'select_federated'}
        max_length = max([len(x) for k, x in select_values.items()])
        for k, v in select_values.items():
            if len(v) == 1:
                v *= max_length
            assert len(v) == max_length, f"Length of {k}: {v} does not match " \
                                         f"max length {max_length} of (metrics, filter_type, threshold, take_high)."
        return values


class IVFilterParam(FederatedStandardFilterParam):
    metrics: List[string_choice({'iv'})] = ['iv']


class StatisticFilterParam(StandardFilterParam):
    metrics: List[statistic_metrics_param(describe=False)] = ["mean"]


class ManualFilterParam(pydantic.BaseModel):
    keep_col: List[str] = []
    left_out_col: List[str] = []

    @pydantic.root_validator(pre=False)
    def no_intersection(cls, values):
        left_out_col = values.get('left_out_col', [])
        keep_col = values.get('keep_col', [])
        intersection = set(left_out_col).intersection(set(keep_col))
        if intersection:
            raise ValueError(f"`keep_col` and `left_out_col` share common elements: {intersection}")
        return values
