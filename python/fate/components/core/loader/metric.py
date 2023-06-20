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
from typing import Union

from fate.interface import InCompleteMetrics, Metrics, MetricsHandler


def load_metrics_handler():
    return ComponentMetricsHandler()


class ComponentMetricsHandler(MetricsHandler):
    """
    this implement use ctx.writer(artifact).write_metric() as metric output sink
    """

    def __init__(self) -> None:
        self._metric_handlers = {}

    def register_metrics(self, **kwargs):
        for name, handler in kwargs.items():
            self._metric_handlers[name] = handler

    def log_metrics(self, metrics: Union[Metrics, InCompleteMetrics]):
        if metrics.name not in self._metric_handlers:
            raise ValueError(f"metric named `{metrics.name}` not registered")
        handler = self._metric_handlers[metrics.name]
        handler.write_metric(metrics)
