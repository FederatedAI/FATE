import hashlib
from dataclasses import dataclass

from .uri import URI, ConcrateURI

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
