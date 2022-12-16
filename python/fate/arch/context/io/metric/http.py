from typing import Union

import requests

from ....unify import URI
from ...metric import InCompleteMetrics, Metrics


class HTTPMetricsWriter:
    def __init__(self, ctx, name: str, uri: URI, metadata) -> None:
        self.name = name
        self.ctx = ctx
        self.entrypoint = f"{uri.schema}://{uri.authority}{uri.path}"

    def write_metric(self, metrics: Union[Metrics, InCompleteMetrics]):
        if isinstance(metrics, Metrics):
            requests.post(url=self.entrypoint, json={"data": metrics.dict(), "incomplte": False})
        else:
            requests.post(url=self.entrypoint, json={"data": metrics.dict(), "incomplete": True})


class HTTPMetricsReader:
    def __init__(self, ctx, name: str, uri: URI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.entrypoint = f"{uri.schema}://{uri.authority}{uri.path}"

    def read_metric(self):
        metric_dict = requests.get(url=self.entrypoint).json().get("data", {})
        return metric_dict
