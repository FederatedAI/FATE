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

from fate.arch.context._metrics import (
    BaseMetricsHandler,
    InMemoryMetricsHandler,
    OneTimeMetrics,
    StepMetrics,
)

from .artifacts.metric import JsonMetricFileWriter, JsonMetricRestfulWriter


class ComponentMetricsFileHandler(InMemoryMetricsHandler):
    def __init__(self, writer: JsonMetricFileWriter) -> None:
        self._writer = writer
        super().__init__()

    def finalize(self):
        self._writer.write(self.get_metrics())


class ComponentMetricsRestfulHandler(BaseMetricsHandler):
    def __init__(self, writer: JsonMetricRestfulWriter) -> None:
        self._writer = writer

    def _log_step_metrics(self, metrics: "StepMetrics"):
        record = metrics.to_record()
        self._writer.write(record.dict())

    def _log_one_time_metrics(self, metrics: "OneTimeMetrics"):
        record = metrics.to_record()
        self._writer.write(record.dict())

    def finalize(self):
        self._writer.close()
