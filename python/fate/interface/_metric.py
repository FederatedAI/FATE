from typing import List, Protocol, Tuple


class Metrics(Protocol):
    name: str
    type: str

    def dict(self) -> dict:
        ...


class Metric(Protocol):
    type: str

    def dict(self) -> dict:
        ...


class MetricsHandler(Protocol):
    def log_metrics(self, metrics: Metrics):
        ...

    def log_metric(self, name: str, metric: Metric, step=None, timestamp=None):
        ...

    # concrate
    def log_scalar(self, name: str, data: float, step=None, timestamp=None):
        ...

    def log_loss(self, name: str, data: float, step, timestamp=None):
        ...

    def log_roc(self, name: str, data: List[Tuple[float, float]]):
        ...

    def log_accuracy(self, name: str, accuracy: float, step, timestamp=None):
        ...
