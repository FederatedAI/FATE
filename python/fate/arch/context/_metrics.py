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

import abc
import time
from typing import Dict, List, Optional, Tuple, Union

from ._namespace import NS, IndexedNS


class NoopMetricsHandler:
    def __init__(self) -> None:
        self._metrics = {}

    def log_intime(self, metrics: "Metrics"):
        pass

    def log_metrics(self, metrics: Union["Metrics", "InCompleteMetrics"]):
        if isinstance(metrics, Metrics):
            if metrics.name in self._metrics:
                raise ValueError(f"duplicated metrics: `{metrics.name}` already exists")
            else:
                self._metrics[metrics.name] = metrics
        elif isinstance(metrics, InCompleteMetrics):
            if (metrics.name, metrics.namespace) not in self._metrics:
                self._metrics[(metrics.name, metrics.namespace)] = metrics
            else:
                self._metrics[(metrics.name, metrics.namespace)] = self._metrics[
                    (metrics.name, metrics.namespace)
                ].merge(metrics)
        else:
            raise ValueError(f"metrics `{metrics}` not allowed")


class Metric(metaclass=abc.ABCMeta):
    type: str

    @abc.abstractmethod
    def dict(self) -> dict:
        ...


class Metrics(metaclass=abc.ABCMeta):
    name: str
    type: str
    nemaspace: Optional[str] = None
    groups: Dict[str, str] = {}

    @abc.abstractmethod
    def dict(self) -> dict:
        ...


class InCompleteMetrics(metaclass=abc.ABCMeta):
    name: str
    type: str
    namespace: Optional[str] = None
    groups: Dict[str, str] = {}

    @abc.abstractmethod
    def dict(self) -> dict:
        ...

    @abc.abstractmethod
    def merge(self, metrics: "InCompleteMetrics") -> "InCompleteMetrics":
        ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d) -> "InCompleteMetrics":
        ...


class ScalarMetric(Metric):
    type = "scalar"

    def __init__(self, scalar) -> None:
        self.scalar = scalar

    def dict(self):
        return self.scalar


class LossMetric(Metric):
    type = "loss"

    def __init__(self, loss) -> None:
        self.loss = loss

    def dict(self) -> dict:
        return self.loss


class AccuracyMetric(Metric):
    type = "accuracy"

    def __init__(self, accuracy) -> None:
        self.accuracy = accuracy

    def dict(self) -> dict:
        return self.accuracy


class AUCMetric(Metric):
    type = "auc"

    def __init__(self, auc) -> None:
        self.auc = auc

    def dict(self) -> dict:
        return self.auc


class ROCMetrics(Metrics):
    type = "roc"

    def __init__(self, name, data) -> None:
        self.name = name
        self.data = data
        self.nemaspace: Optional[str] = None
        self.groups: Dict[str, str] = {}

    def dict(self) -> dict:
        return dict(
            name=self.name,
            namespace=self.nemaspace,
            groups=self.groups,
            type=self.type,
            metadata={},
            data=self.data,
        )


class MetricsWrap:
    def __init__(self, handler, namespace: NS) -> None:
        self.handler = handler
        self.namespace = namespace

    def log_metrics(self, metrics: Union[Metrics, InCompleteMetrics]):
        return self.handler.log_metrics(metrics)

    def log_meta(self, meta):
        return self.log_metrics(meta)

    def log_metric(self, name: str, metric: Metric, step=None, timestamp=None, metadata=None):
        if metadata is None:
            metadata = {}
        if step is not None:
            if isinstance(self.namespace, IndexedNS):
                step = self.namespace.index
        if timestamp is None:
            timestamp = time.time()
        return self.log_metrics(
            StepMetrics(
                name=name,
                namespace=self.namespace.get_metrics_keys(),
                type=metric.type,
                data=[dict(metric=metric.dict(), step=step, timestamp=timestamp)],
                metadata=metadata,
            )
        )

    def log_scalar(self, name: str, metric: float, step=None, timestamp=None):
        return self.log_metric(name, ScalarMetric(metric), step, timestamp)

    def log_loss(self, name: str, loss: float, step, timestamp=None):
        return self.log_metric(name, LossMetric(loss), step, timestamp)

    def log_accuracy(self, name: str, accuracy: float, step=None, timestamp=None):
        return self.log_metric(name, AccuracyMetric(accuracy), step, timestamp)

    def log_auc(self, name: str, auc: float, step=None, timestamp=None):
        return self.log_metric(name, AUCMetric(auc), step, timestamp)

    def log_roc(self, name: str, data: List[Tuple[float, float]]):
        return self.log_metrics(ROCMetrics(name, data))


class StepMetrics(InCompleteMetrics):
    complete = False

    def __init__(self, name, namespace, type, data, metadata) -> None:
        self.name = name
        self.namespace = namespace
        self.type = type
        self.data = data
        self.metadata = metadata

    def merge(self, metrics: InCompleteMetrics):
        if not isinstance(metrics, StepMetrics):
            raise ValueError(f"can't merge metrics type `{metrics}` with StepMetrics")
        if metrics.type != self.type:
            raise ValueError(f"can't merge metrics type `{metrics}` with StepMetrics named `{self.name}`")
        if metrics.namespace != self.namespace:
            raise ValueError(
                f"can't merge metrics namespace `{self.namespace}` with `{metrics.namespace}` named `{self.name}`"
            )

        return StepMetrics(
            name=self.name,
            type=self.type,
            namespace=self.namespace,
            data=[*self.data, *metrics.data],
            metadata=self.metadata,
        )

    def dict(self) -> dict:
        return dict(
            name=self.name,
            namespace=self.namespace,
            type=self.type,
            metadata=self.metadata,
            data=self.data,
        )

    @classmethod
    def from_dict(cls, d) -> "StepMetrics":
        return StepMetrics(**d)

    def __str__(self):
        return f"StepMetrics(name={self.name}, type={self.type}, namespace={self.namespace}, data={self.data}, metadata={self.metadata})"

    def __repr__(self):
        return self.__str__()
