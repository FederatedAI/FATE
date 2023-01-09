#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
