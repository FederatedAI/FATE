from typing import List, Optional, Protocol


class Metric(Protocol):
    key: str
    value: float
    timestamp: Optional[float] = None


class MetricMeta(Protocol):
    name: str
    metric_type: str
    metas: dict
    extra_metas: Optional[dict] = None

    def update_metas(self, metas: dict):
        ...


class Metrics(Protocol):
    def log(self, name: str, namespace: str, data: List[Metric]):
        ...

    def log_meta(self, name: str, namespace: str, meta: MetricMeta):
        ...

    def log_warmstart_init_iter(self, iter_num):  # FIXME: strange here
        ...
