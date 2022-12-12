from typing import List, Tuple

from fate.interface import Metric, Metrics, MetricsHandler


class ScalarMetric(Metric):
    type = "scalar"

    def __init__(self, scalar) -> None:
        self.scalar = scalar

    def dict(self):
        return self.scalar


class LossMetric(Metric):
    type = "loss"

    def __init__(self, loss) -> None:
        self.loss = loss

    def dict(self) -> dict:
        return self.loss


class AccuracyMetric(Metric):
    type = "accuracy"

    def __init__(self, accuracy) -> None:
        self.accuracy = accuracy

    def dict(self) -> dict:
        return self.accuracy


class StepMetrics(Metrics):
    def __init__(self, name, type: str) -> None:
        self.name = name
        self.type = type
        self.data = []

    def add(self, metric, step, timestamp):
        self.data.append((metric, step, timestamp))

    def dict(self) -> dict:
        return dict(
            name=self.name,
            type=self.type,
            data=[dict(metric=item[0].dict(), step=item[1], timestamp=item[2]) for item in self.data],
        )


class ROCMetrics(Metrics):
    type = "roc"

    def __init__(self, name, data) -> None:
        self.name = name
        self.data = data

    def dict(self) -> dict:
        return dict(
            name=self.name,
            type=self.type,
            data=self.data,
        )


class NoopMetricsHandler(MetricsHandler):
    def __init__(self) -> None:
        self._metrics = {}

    # general
    def log_metrics(self, metrics: Metrics):
        if metrics.name in self._metrics:
            raise ValueError(f"duplicated metircs: `{metrics.name}` already exists")
        self._metrics[metrics.name] = metrics

    def log_metric(self, name: str, metric: Metric, step=None, timestamp=None):
        if name not in self._metrics:
            self._metrics[name] = StepMetrics(name, metric.type)
        metircs = self._metrics[name]
        if isinstance(metircs, StepMetrics) and metircs.type == metric.type:
            self._metrics[name].add(metric, step, timestamp)
        else:
            raise ValueError(f"expected step metrics with type {metircs.type}, got `{type(metric)}`")

    # concrate
    def log_scalar(self, name: str, metric: float, step=None, timestamp=None):
        return self.log_metric(name, ScalarMetric(metric), step, timestamp)

    def log_loss(self, name: str, loss: float, step, timestamp=None):
        return self.log_metric(name, LossMetric(loss), step, timestamp)

    def log_accuracy(self, name: str, accuracy: float, step, timestamp=None):
        return self.log_metric(name, AccuracyMetric(accuracy), step, timestamp)

    def log_roc(self, name: str, data: List[Tuple[float, float]]):
        return self.log_metrics(ROCMetrics(name, data))
