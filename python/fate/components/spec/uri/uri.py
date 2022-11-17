import logging
import re
from dataclasses import dataclass
from typing import Dict, MutableMapping, Optional, Protocol, Type

# see https://www.rfc-editor.org/rfc/rfc3986#appendix-B
# scheme    = $2
# authority = $4
# path      = $5
# query     = $7
# fragment  = $9
_URI_REGEX = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")

logger = logging.getLogger(__name__)


class _PROTOCOLS:
    cache: Optional[MutableMapping[str, Type["ConcrateURI"]]] = None

    @classmethod
    def get_concrate(cls, schema: str):
        # update schema
        if cls.cache is None:
            cls.cache = {}
            # buildin protocol
            from .eggroll import EggrollURI
            from .file import FileURI
            from .flow import FlowURI
            from .hdfs import HdfsURI

            cls.cache[EggrollURI.schema()] = EggrollURI
            cls.cache[HdfsURI.schema()] = HdfsURI
            cls.cache[FileURI.schema()] = FileURI
            cls.cache[FlowURI.schema()] = FlowURI

            # register from entrypoint
            import pkg_resources

            for concrate_uri_ep in pkg_resources.iter_entry_points(group="fate.plugins.uri"):
                try:
                    concrate_uri = concrate_uri_ep.load()
                    concrate_uri_schema = concrate_uri.schema()
                except Exception:
                    logger.warning(
                        f"register uri schema from entrypoint(named={concrate_uri_ep.name}, module={concrate_uri_ep.module_name}) failed"
                    )
                    continue
                try:
                    assert hasattr(concrate_uri, "from_uri"), f"bad implement of uri for schema: {concrate_uri_schema}"
                except Exception as e:
                    logger.warning(f"register uri schema {concrate_uri_schema} from entrypoint failed: {e}")

        if schema not in cls.cache:
            raise ValueError(f"schema `{schema}` not supported, use one of {list(cls.cache)}")
        return cls.cache[schema]


@dataclass
class URI:
    schema: str
    path: str
    query: Optional[str] = None
    fragment: Optional[str] = None
    authority: Optional[str] = None

    @classmethod
    def option(cls, key, value):
        ...

    @classmethod
    def from_string(cls, uri_str: str) -> "ConcrateURI":
        match = _URI_REGEX.fullmatch(uri_str)
        if match is None:
            raise ValueError(f"`{uri_str}` is not valid uri")
        _, schema, _, authority, path, _, query, _, fragment = match.groups()
        uri = URI(schema, path, query, fragment, authority)
        return _PROTOCOLS.get_concrate(uri.schema).from_uri(uri)


class ConcrateURI(Protocol):
    @classmethod
    def schema(cls) -> str:
        ...

    @classmethod
    def from_uri(cls, uri: URI) -> "ConcrateURI":
        ...

    def read_json(self) -> Dict:
        ...

    def read_yaml(self) -> Dict:
        ...

    def read_kv(self, key: str):
        ...

    def write_kv(self, key: str, value):
        ...

    def read_df(self, ctx):
        ...
