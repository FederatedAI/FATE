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

from fate.interface import MetricsHandler

from ._type import InCompleteMetrics, Metrics


class NoopMetricsHandler(MetricsHandler):
    def __init__(self) -> None:
        self._metrics = {}

    def log_metrics(self, metrics: Union[Metrics, InCompleteMetrics]):
        if isinstance(metrics, Metrics):
            if metrics.name in self._metrics:
                raise ValueError(f"duplicated metircs: `{metrics.name}` already exists")
            else:
                self._metrics[metrics.name] = metrics
        elif isinstance(metrics, InCompleteMetrics):
            if metrics.name not in self._metrics:
                self._metrics[metrics.name] = metrics
            else:
                self._metrics[metrics.name].merge(metrics)
        else:
            raise ValueError(f"metrics `{metrics}` not allowed")
