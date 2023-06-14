import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Generic, TypeVar

from fate.arch.context.io.data.df import Dataframe

W = TypeVar("W")
T = TypeVar("T")


class Slot(Generic[W]):
    def __init__(self, writer: W) -> None:
        self._writer = writer

    @contextmanager
    def writer(self) -> Generator[W, Any, Any]:
        yield self._writer


class Slots(Generic[W]):
    def __init__(self, writer_generator) -> None:
        self._writer_generator = writer_generator

    @contextmanager
    def writer(self, index) -> Generator[W, Any, Any]:
        yield self._writer_generator.get_writer(index)


class SlotWriter(Generic[T]):
    def write(self, slot: T):
        ...


class LocalFileWriter:
    def __init__(self, path: Path) -> None:
        self._path = path

    def get_file_path(self) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return self._path


class JsonWriter(SlotWriter[dict]):
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, slot: dict):
        with self._path.open("w") as fw:
            json.dump(slot, fw)
