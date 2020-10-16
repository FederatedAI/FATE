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
#
from enum import Enum


class MetricType(Enum):
    LOSS = 'LOSS'


class Metric(object):
    def __init__(self, key, value: float, timestamp: float = None):
        self.key = key
        self.value = value
        self.timestamp = timestamp


class MetricMeta(object):
    def __init__(self, name: str, metric_type: MetricType, extra_metas: dict = None):
        self.name = name
        self.metric_type = metric_type
        self.metas = {}
        if extra_metas:
            self.metas.update(extra_metas)
        self.metas['name'] = name
        self.metas['metric_type'] = metric_type

    def update_metas(self, metas: dict):
        self.metas.update(metas)

    def to_dict(self):
        return self.metas


