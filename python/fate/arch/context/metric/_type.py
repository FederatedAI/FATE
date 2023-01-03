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
