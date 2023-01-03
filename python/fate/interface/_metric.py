from typing import List, Protocol, Tuple, Union


class Metric(Protocol):
    type: str

    def dict(self) -> dict:
        ...


class Metrics(Protocol):
    name: str
    type: str

    def dict(self) -> dict:
        ...


class InCompleteMetrics(Protocol):
    name: str
    type: str

    def dict(self) -> dict:
        ...

    def merge(self, metrics: "InCompleteMetrics"):
        ...

    @classmethod
    def from_dict(cls, d) -> "InCompleteMetrics":
        ...


class MetricsHandler(Protocol):
    def log_metrics(self, metrics: Union[Metrics, InCompleteMetrics]):
        ...


class MetricsWrap(Protocol):
    def into_group(self, group_name: str, group_id: str) -> "MetricsWrap":
        ...

    def log_metrics(self, metrics: Metrics):
        ...

    def log_meta(self, meta):
        ...

    def log_metric(self, name: str, metric: Metric, step=None, timestamp=None):
        ...

    def log_scalar(self, name: str, metric: float, step=None, timestamp=None):
        ...

    def log_loss(self, name: str, loss: float, step, timestamp=None):
        ...

    def log_accuracy(self, name: str, accuracy: float, step, timestamp=None):
        ...

    def log_roc(self, name: str, data: List[Tuple[float, float]]):
        ...
