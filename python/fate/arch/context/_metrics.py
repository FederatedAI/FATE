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

import time
import typing
from typing import Dict, List, Optional, Tuple, Union

import pydantic

from ._namespace import NS, IndexedNS


class BaseMetricsHandler:
    def log_metrics(self, metrics: Union["StepMetrics", "OneTimeMetrics"]):
        if isinstance(metrics, StepMetrics):
            self._log_step_metrics(metrics)
        elif isinstance(metrics, OneTimeMetrics):
            self._log_one_time_metrics(metrics)
        else:
            raise ValueError(f"metrics `{metrics}` not allowed")

    def _log_step_metrics(self, metrics: "StepMetrics"):
        raise NotImplementedError

    def _log_one_time_metrics(self, metrics: "OneTimeMetrics"):
        raise NotImplementedError


class InMemoryMetricsHandler(BaseMetricsHandler):
    def __init__(self):
        self._step_metrics: typing.Dict[typing.Any, "StepMetrics"] = {}
        self._one_time_metrics: typing.Dict[typing.Any, "OneTimeMetrics"] = {}

    def _log_step_metrics(self, metrics: "StepMetrics"):
        if (metrics.name, tuple(metrics.groups)) in self._step_metrics:
            self._step_metrics[(metrics.name, tuple(metrics.groups))] = self._step_metrics[
                (metrics.name, tuple(metrics.groups))
            ].merge(metrics)
        else:
            self._step_metrics[(metrics.name, tuple(metrics.groups))] = metrics

    def _log_one_time_metrics(self, metrics: "OneTimeMetrics"):
        if (metrics.name, tuple(metrics.groups)) in self._one_time_metrics:
            raise ValueError(f"duplicated metrics: `{metrics.name}` already exists")
        else:
            self._one_time_metrics[(metrics.name, tuple(metrics.groups))] = metrics

    def get_metrics(self):
        metrics = []
        for k, v in self._step_metrics.items():
            metrics.append(v.dict())
        for k, v in self._one_time_metrics.items():
            metrics.append(v.dict())
        return metrics


class MetricsWrap:
    def __init__(self, handler, namespace: NS) -> None:
        self.handler = handler
        self.namespace = namespace

    def log_metrics(self, data, name: str, type: Optional[str] = None):
        return self.handler.log_metrics(
            OneTimeMetrics(
                name=name,
                type=type,
                groups=[*self.namespace.metric_groups, self.namespace.get_group()],
                data=data,
            )
        )

    def log_step(self, value, name: str, type: Optional[str] = None):
        if isinstance(self.namespace, IndexedNS):
            step = self.namespace.index
        else:
            raise RuntimeError(
                "log step metric only allowed in indexed namespace since the step is inferred from namespace"
            )
        timestamp = time.time()
        return self.handler.log_metrics(
            StepMetrics(
                name=name,
                type=type,
                groups=self.namespace.metric_groups,
                step_axis=self.namespace.name,
                data=[dict(metric=value, step=step, timestamp=timestamp)],
            )
        )

    def log_scalar(self, name: str, scalar: float):
        return self.log_step(value=scalar, name=name, type="scalar")

    def log_loss(self, name: str, loss: float):
        return self.log_step(value=loss, name=name, type="loss")

    def log_accuracy(self, name: str, accuracy: float):
        return self.log_step(value=accuracy, name=name, type="accuracy")

    def log_auc(self, name: str, auc: float):
        return self.log_step(value=auc, name=name, type="auc")

    def log_roc(self, name: str, data: List[Tuple[float, float]]):
        return self.log_metrics(data=data, name=name, type="roc")


class OneTimeMetrics:
    def __init__(
        self, name: str, type: Optional[str], groups: List[Tuple[str, Optional[int]]], data: Union[List, Dict]
    ) -> None:
        self.name = name
        self.groups = groups
        self.type = type
        self.data = data

    def dict(self):
        return self.to_record().dict()

    def to_record(self):
        return MetricRecord(
            name=self.name,
            groups=[MetricRecord.Group(name=k, index=v) for k, v in self.groups],
            type=self.type,
            step_axis=None,
            data=self.data,
        )

    def __str__(self):
        return str(self.dict())

    def __repr__(self):
        return self.__str__()


class StepMetrics:
    def __init__(
        self, name: str, type: Optional[str], groups: List[Tuple[str, Optional[int]]], step_axis: str, data: List
    ) -> None:
        self.name = name
        self.type = type
        self.groups = groups
        self.step_axis = step_axis
        self.data = data

    def merge(self, metrics: "StepMetrics"):
        if (
            isinstance(metrics, StepMetrics)
            and metrics.type == self.type
            and metrics.name == self.name
            and metrics.step_axis == self.step_axis
            and metrics.groups == self.groups
        ):
            return StepMetrics(
                name=self.name,
                type=self.type,
                groups=self.groups,
                step_axis=self.step_axis,
                data=[*self.data, *metrics.data],
            )

        raise RuntimeError(f"metrics merge not allowed: `{metrics}` with `{self}`")

    def dict(self) -> dict:
        return self.to_record().dict()

    def to_record(self) -> "MetricRecord":
        return MetricRecord(
            name=self.name,
            type=self.type,
            groups=[MetricRecord.Group(name=g[0], index=g[1]) for g in self.groups],
            step_axis=self.step_axis,
            data=self.data,
        )

    def __str__(self):
        return f"StepMetrics(name={self.name}, type={self.type}, groups={self.groups}, step_axis={self.step_axis}, data={self.data})"

    def __repr__(self):
        return self.__str__()


class MetricRecord(pydantic.BaseModel):
    class Group(pydantic.BaseModel):
        name: str
        index: Optional[int]

    name: str
    type: Optional[str]
    groups: List[Group]
    step_axis: Optional[str]
    data: Union[List, Dict]
