import json
import os
from typing import Union

from ....unify import URI
from ...metric import InCompleteMetrics, Metrics


class FileMetricsWriter:
    def __init__(self, ctx, name: str, uri: URI, metadata) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri

    def write_metric(self, metrics: Union[Metrics, InCompleteMetrics]):
        if isinstance(metrics, Metrics):
            with open(self.uri.path, "w") as f:
                json.dump(metrics.dict(), f)
        else:
            # read
            if not os.path.exists(self.uri.path):
                merged = metrics
            else:
                with open(self.uri.path, "r") as f:
                    merged = metrics.from_dict(json.load(f)).merge(metrics)

            with open(self.uri.path, "w") as f:
                json.dump(merged.dict(), f)


class FileMetricsReader:
    def __init__(self, ctx, name: str, uri: URI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.metadata = metadata

    def read_metric(self):
        with open(self.uri.path, "r") as fin:
            metric_dict = json.loads(fin.read())
        return metric_dict
