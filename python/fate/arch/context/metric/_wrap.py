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
from typing import List, Optional, Tuple, Union

from fate.interface import MetricsHandler
from fate.interface import MetricsWrap as MetricsWrapProtocol

from ._handler import NoopMetricsHandler
from ._incomplte_metrics import StepMetrics
from ._metric import AccuracyMetric, AUCMetric, LossMetric, ScalarMetric
from ._metrics import ROCMetrics
from ._type import InCompleteMetrics, Metric, Metrics


class MetricsWrap(MetricsWrapProtocol):
    def __init__(self, handler: Optional[MetricsHandler], groups=None) -> None:
        if handler is None:
            self.handler = NoopMetricsHandler()
        else:
            self.handler = handler
        if groups is None:
            self.groups = {}
        else:
            self.groups = groups

    def into_group(self, group_name: str, group_id: str) -> "MetricsWrap":
        if group_name in self.groups:
            raise ValueError(
                f"can't into group named `{group_name}` since `{group_name}` already in groups `{self.groups}`"
            )
        groups = {**self.groups}
        groups.update({group_name: group_id})
        return MetricsWrap(self.handler, groups)

    def log_metrics(self, metrics: Union[Metrics, InCompleteMetrics]):
        if self.groups:
            for group_name, group_id in self.groups.items():
                if group_name in metrics.groups:
                    if (to_add_group_id := metrics.groups[group_name]) != group_id:
                        raise ValueError(
                            f"try to add group named `{group_name}`, but group id `{group_id}` not equals `{to_add_group_id}`"
                        )
                else:
                    metrics.groups[group_name] = group_id
        return self.handler.log_metrics(metrics)

    def log_meta(self, meta):
        return self.log_metrics(meta)

    def log_metric(
        self, name: str, metric: Metric, step=None, timestamp=None, namespace=None, groups=None, metadata=None
    ):
        if groups is None:
            groups = {}
        if metadata is None:
            metadata = {}
        return self.log_metrics(
            StepMetrics(
                name=name,
                type=metric.type,
                namespace=namespace,
                groups=groups,
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
