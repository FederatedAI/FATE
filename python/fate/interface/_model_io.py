import contextlib
from typing import Iterator, List, Protocol


class ModelMeta:
    meta_type: str
    meta_version: str
    meta_data: dict


class ModelWriter(Protocol):
    def write_meta(self, meta: ModelMeta):
        ...

    def write_bytes(self, buffer):
        ...


class ModelsSaver(Protocol):
    @contextlib.contextmanager
    def with_name(self, name) -> Iterator[ModelWriter]:
        ...


class ModelReader(Protocol):
    def read_meta(self) -> ModelMeta:
        ...

    def read_bytes(self):
        ...


class ModelsLoader(Protocol):
    names: List
    has_model: bool
    has_isometric_model: bool

    @contextlib.contextmanager
    def with_name(self, name) -> Iterator[ModelReader]:
        ...
