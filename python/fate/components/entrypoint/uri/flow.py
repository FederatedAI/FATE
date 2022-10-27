from dataclasses import dataclass
from typing import Dict

import requests

from .uri import URI, ConcrateURI


@dataclass
class FlowURI(ConcrateURI):
    url: str

    @classmethod
    def schema(cls):
        return "flow"

    @classmethod
    def from_uri(cls, uri: URI):
        return FlowURI(f"http://{uri.authority}{uri.path}")

    def read_json(self) -> Dict:
        return requests.get(self.url).json()

    def write_json(self, json: Dict):
        return requests.post(self.url, json=json)

    def read_kv(self, key: str):
        return requests.get(self.url, params={"key": key}).json()[key]

    def write_kv(self, key: str, value):
        return requests.post(self.url, json={key: value})
