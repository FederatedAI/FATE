from typing import List, Optional, Tuple

from fate.interface import MetricsHandler

from ._handler import NoopMetricsHandler
from ._incomplte_metrics import StepMetrics
from ._metric import AccuracyMetric, LossMetric, ScalarMetric
from ._metrics import ROCMetrics
from ._type import Metric, Metrics


class MetricsWrap:
    def __init__(self, handler: Optional[MetricsHandler]) -> None:
        if handler is None:
            self.handler = NoopMetricsHandler()
        else:
            self.handler = handler

    def log_metrics(self, metrics: Metrics):
        return self.handler.log_metrics(metrics)

    def log_metric(self, name: str, metric: Metric, step=None, timestamp=None):
        return self.handler.log_metrics(StepMetrics.from_step_metric(name, metric, step, timestamp))

    def log_scalar(self, name: str, metric: float, step=None, timestamp=None):
        return self.log_metric(name, ScalarMetric(metric), step, timestamp)

    def log_loss(self, name: str, loss: float, step, timestamp=None):
        return self.log_metric(name, LossMetric(loss), step, timestamp)

    def log_accuracy(self, name: str, accuracy: float, step, timestamp=None):
        return self.log_metric(name, AccuracyMetric(accuracy), step, timestamp)

    def log_roc(self, name: str, data: List[Tuple[float, float]]):
        return self.log_metrics(ROCMetrics(name, data))
