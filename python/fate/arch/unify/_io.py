import hashlib
import re
from abc import ABCMeta
from dataclasses import dataclass
from typing import Optional

# see https://www.rfc-editor.org/rfc/rfc3986#appendix-B
# scheme    = $2
# authority = $4
# path      = $5
# query     = $7
# fragment  = $9
_uri_regex = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")


@dataclass
class URI:
    schema: str
    path: str
    query: Optional[str] = None
    fragment: Optional[str] = None
    authority: Optional[str] = None

    @classmethod
    def from_string(cls, uri: str) -> "URI":
        match = _uri_regex.fullmatch(uri)
        if match is None:
            raise ValueError(f"`{uri}` is not valid uri")
        _, schema, _, authority, path, _, query, _, fragment = match.groups()
        return URI(schema, path, query, fragment, authority)

    def to_schema(self):
        for cls in ConcrateURI.__subclasses__():
            if cls.schema() == self.schema:
                return cls.from_uri(self)


class ConcrateURI(metaclass=ABCMeta):
    @classmethod
    def schema(cls) -> str:
        ...

    @classmethod
    def from_uri(cls, uri: URI) -> "ConcrateURI":
        ...


_EGGROLL_NAME_MAX_SIZE = 128


@dataclass
class EggrollURI(ConcrateURI):
    namespace: str
    name: str

    @classmethod
    def schema(cls):
        return "eggroll"

    @classmethod
    def from_uri(cls, uri: URI):
        _, namespace, *names = uri.path.split("/")
        name = "_".join(names)
        if len(name) > _EGGROLL_NAME_MAX_SIZE:
            name = hashlib.md5(name.encode(encoding="utf8")).hexdigest()[
                :_EGGROLL_NAME_MAX_SIZE
            ]
        return EggrollURI(namespace, name)


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


@dataclass
class HttpURI(ConcrateURI):
    path: str
    authority: Optional[str] = None



    @classmethod
    def schema(cls):
        return "http"

    @classmethod
    def from_uri(cls, uri: URI):
        return HttpURI(uri.path, uri.authority)


@dataclass
class HttpsURI(ConcrateURI):
    path: str
    authority: Optional[str] = None

    @classmethod
    def schema(cls):
        return "https"

    @classmethod
    def from_uri(cls, uri: URI):
        return HttpsURI(uri.path, uri.authority)


if __name__ == "__main__":
    print(URI.from_string("file:///aaa"))
    print(URI.from_string("eggroll:///namespace/name1/name2").to_schema())
    print(URI.from_string("eggroll:///namespace/name1/name2"))
    print(URI.from_string("http://127.0.0.1:9999/test"))
    print(1)
    print(URI.from_string("http://127.0.0.1:9999/test").full_path)
    print(URI.from_string("https://127.0.0.1:9999/test"))
    print(URI.from_string("hdsf://127.0.0.1:9999/namespace/name"))
    print(URI.from_string("hive://sage:pass@127.0.0.1:9999/namespace/name"))
