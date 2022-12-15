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
