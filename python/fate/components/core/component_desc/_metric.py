from typing import Dict, List, Optional, Union

import pydantic
from fate.arch.context._metrics import InMemoryMetricsHandler
from fate.components.core.component_desc.artifacts.metric import JsonMetricFileWriter


class ComponentMetricsHandler(InMemoryMetricsHandler):
    def __init__(self, writer: JsonMetricFileWriter) -> None:
        self._writer = writer
        super().__init__()

    def finalize(self):
        self._writer.write(self.get_metrics())
