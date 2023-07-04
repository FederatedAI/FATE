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
from typing import List, Optional, Tuple, Union

from ._namespace import NS, IndexedNS


class NoopMetricsHandler:
    def __init__(self) -> None:
        self._metrics = {}

    def __contains__(self, item: Union["StepMetrics", "OneTimeMetrics"]):
        if not isinstance(item, (StepMetrics, OneTimeMetrics)):
            return False
        return (item.name, item.namespaces, item.groups) in self._metrics

    def __getitem__(self, item: Union["StepMetrics", "OneTimeMetrics"]):
        if not isinstance(item, (StepMetrics, OneTimeMetrics)):
            raise ValueError(f"metrics `{item}` not allowed")
        return self._metrics[(item.name, item.namespaces, item.groups)]

    def __setitem__(self, key: Union["StepMetrics", "OneTimeMetrics"], value: Union["StepMetrics", "OneTimeMetrics"]):
        if not isinstance(key, (StepMetrics, OneTimeMetrics)):
            raise ValueError(f"metrics `{key}` not allowed")
        if not isinstance(value, (StepMetrics, OneTimeMetrics)):
            raise ValueError(f"metrics `{value}` not allowed")
        self._metrics[(key.name, key.namespaces, key.groups)] = value

    def log_metrics(self, metrics: Union["StepMetrics", "OneTimeMetrics"]):
        if isinstance(metrics, StepMetrics):
            if metrics in self:
                self[metrics] = self[metrics].merge(metrics)
            else:
                self[metrics] = metrics
        elif isinstance(metrics, OneTimeMetrics):
            if metrics in self:
                raise ValueError(f"duplicated metrics: `{metrics.name}` already exists")
            else:
                self[metrics] = metrics
        else:
            raise ValueError(f"metrics `{metrics}` not allowed")


class MetricsWrap:
    def __init__(self, handler, namespace: NS) -> None:
        self.handler = handler
        self.namespace = namespace

    def log_metrics(self, values, name: str, type: str, metadata: Optional[dict] = None):
        if metadata is None:
            metadata = {}
        return self.handler.log_metrics(
            OneTimeMetrics(
                name=name,
                namespaces=self.namespace.get_metrics_keys().namespaces,
                groups=self.namespace.get_metrics_keys().groups,
                type=type,
                data=values,
                metadata=metadata,
            )
        )

    def log_metric(self, value, name: str, type: str, step=None, timestamp=None, metadata: Optional[dict] = None):
        if metadata is None:
            metadata = {}
        if step is None:
            if isinstance(self.namespace, IndexedNS):
                step = self.namespace.index
        if timestamp is None:
            timestamp = time.time()
        metric_key = self.namespace.get_metrics_keys()
        return self.handler.log_metrics(
            StepMetrics(
                name=name,
                namespaces=metric_key.namespaces,
                groups=metric_key.groups,
                type=type,
                data=[dict(metric=value, step=step, timestamp=timestamp)],
                metadata=metadata,
            )
        )

    def log_scalar(self, name: str, scalar: float, step=None, timestamp=None):
        return self.log_metric(value=scalar, name=name, type="scalar", step=step, timestamp=timestamp)

    def log_loss(self, name: str, loss: float, step, timestamp=None):
        return self.log_metric(value=loss, name=name, type="loss", step=step, timestamp=timestamp)

    def log_accuracy(self, name: str, accuracy: float, step=None, timestamp=None):
        return self.log_metric(value=accuracy, name=name, type="accuracy", step=step, timestamp=timestamp)

    def log_auc(self, name: str, auc: float, step=None, timestamp=None):
        return self.log_metric(value=auc, name=name, type="auc", step=step, timestamp=timestamp)

    def log_roc(self, name: str, data: List[Tuple[float, float]], metadata=None):
        if metadata is None:
            metadata = {}
        return self.log_metrics(
            OneTimeMetrics(
                name=name,
                namespaces=self.namespace.get_metrics_keys().namespaces,
                groups=self.namespace.get_metrics_keys().groups,
                type="roc",
                data=data,
                metadata=metadata,
            )
        )


class OneTimeMetrics:
    def __init__(self, name, namespaces, groups, type, data, metadata) -> None:
        self.name = name
        self.namespaces = namespaces
        self.groups = groups
        self.type = type
        self.data = data
        self.metadata = metadata

    def dict(self):
        return dict(
            name=self.name,
            namespace=self.namespaces,
            groups=self.groups,
            type=self.type,
            metadata=self.metadata,
            data=self.data,
        )

    def __str__(self):
        return str(self.dict())

    def __repr__(self):
        return self.__str__()


class StepMetrics:
    complete = False

    def __init__(self, name, namespaces, groups, type, data, metadata) -> None:
        self.name = name
        self.namespaces = namespaces
        self.groups = groups
        self.type = type
        self.data = data
        self.metadata = metadata

    def merge(self, metrics: "StepMetrics"):
        if not isinstance(metrics, StepMetrics):
            raise ValueError(f"can't merge metrics type `{metrics}` with StepMetrics")
        if metrics.type != self.type:
            raise ValueError(f"can't merge metrics type `{metrics}` with StepMetrics named `{self.name}`")
        if metrics.namespaces != self.namespaces:
            raise ValueError(
                f"can't merge metrics namespace `{self.namespaces}` with `{metrics.namespaces}` named `{self.name}`"
            )
        if metrics.groups != self.groups:
            raise ValueError(f"can't merge metrics groups `{self.groups}` with `{metrics.groups}` named `{self.name}`")

        return StepMetrics(
            name=self.name,
            namespaces=self.namespaces,
            groups=self.groups,
            type=self.type,
            data=[*self.data, *metrics.data],
            metadata=self.metadata,
        )

    def dict(self) -> dict:
        return dict(
            name=self.name,
            namespaces=self.namespaces,
            groups=self.groups,
            type=self.type,
            metadata=self.metadata,
            data=self.data,
        )

    @classmethod
    def from_dict(cls, d) -> "StepMetrics":
        return StepMetrics(**d)

    def __str__(self):
        return f"StepMetrics(name={self.name}, namespaces={self.namespaces}, groups={self.groups}, type={self.type}, metadata={self.metadata}, data={self.data})"

    def __repr__(self):
        return self.__str__()
