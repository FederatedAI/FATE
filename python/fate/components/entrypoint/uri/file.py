import json
import pathlib
from dataclasses import dataclass
from typing import Dict

from .uri import URI, ConcrateURI


@dataclass
class FileURI(ConcrateURI):
    path: str

    @classmethod
    def schema(cls):
        return "file"

    @classmethod
    def from_uri(cls, uri: URI):
        pathlib.Path(uri.path).parent.mkdir(parents=True, exist_ok=True)
        return FileURI(uri.path)

    def read_json(self) -> Dict:
        with open(self.path) as f:
            return json.load(f)

    def read_kv(self, key: str):
        with open(self.path) as f:
            json.load(f)[key]

    def write_kv(self, key: str, value):
        with open(self.path, "w") as f:
            json.dump({key: value}, f)

    def write_json(self, d: Dict):
        with open(self.path, "w") as f:
            json.dump(d, f)

    def read_df(self, ctx):
        ...
