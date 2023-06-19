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

from fate.interface import MetricsHandler
from fate.interface import MetricsWrap as MetricsWrapProtocol

from .._namespace import NS, IndexedNS
from ._incomplte_metrics import StepMetrics
from ._metric import AccuracyMetric, AUCMetric, LossMetric, ScalarMetric
from ._metrics import ROCMetrics
from ._type import InCompleteMetrics, Metric, Metrics


class MetricsWrap(MetricsWrapProtocol):
    def __init__(self, handler: MetricsHandler, namespace: NS) -> None:
        self.namespace = namespace
        self.handler = handler

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
