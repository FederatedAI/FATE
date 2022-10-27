from dataclasses import dataclass
from typing import Optional

from .uri import URI, ConcrateURI


@dataclass
class HdfsURI(ConcrateURI):
    path: str
    authority: Optional[str] = None

    @classmethod
    def schema(cls):
        return "hdfs"

    @classmethod
    def from_uri(cls, uri: URI):
        return HdfsURI(uri.path, uri.authority)
