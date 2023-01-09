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
import abc
from typing import Dict, Optional


class Metric(metaclass=abc.ABCMeta):
    type: str

    @abc.abstractmethod
    def dict(self) -> dict:
        ...


class Metrics(metaclass=abc.ABCMeta):
    name: str
    type: str
    nemaspace: Optional[str] = None
    groups: Dict[str, str] = {}

    @abc.abstractmethod
    def dict(self) -> dict:
        ...


class InCompleteMetrics(metaclass=abc.ABCMeta):
    name: str
    type: str
    nemaspace: Optional[str] = None
    groups: Dict[str, str] = {}

    @abc.abstractmethod
    def dict(self) -> dict:
        ...

    @abc.abstractmethod
    def merge(self, metrics: "InCompleteMetrics") -> "InCompleteMetrics":
        ...

    @abc.abstractclassmethod
    def from_dict(cls, d) -> "InCompleteMetrics":
        ...
