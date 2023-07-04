from typing import Dict, List, Optional, Union

import pydantic
from fate.arch.context._metrics import NoopMetricsHandler
from fate.components.core.component_desc.artifacts.metric import JsonMetricWriter


class ComponentMetricsHandler(NoopMetricsHandler):
    def __init__(self, writer: JsonMetricWriter) -> None:
        self._writer = writer
        super().__init__()

    def finalize(self):
        jsonable_metrics = []
        for k, v in self._metrics.items():
            jsonable_metrics.append(
                MetricData(
                    namespace=".".join(v.namespaces),
                    name=v.name,
                    groups=".".join(v.groups),
                    type=v.type,
                    data=v.data,
                ).dict()
            )
        self._writer.write(jsonable_metrics)


class MetricData(pydantic.BaseModel):
    namespace: Optional[str] = None
    name: str
    type: str
    groups: str
    metadata: Dict[str, str] = {}
    data: Union[List, Dict]
