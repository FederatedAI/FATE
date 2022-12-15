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
