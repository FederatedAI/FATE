import contextlib
import typing
from collections.abc import Iterator
from dataclasses import dataclass

from fate.interface import (
    ModelMeta,
    ModelReader,
    ModelsLoader,
    ModelsSaver,
    ModelWriter,
)


@dataclass
class PBModelMeta(ModelMeta):
    meta_data: dict
    meta_type: str = "protobuf"
    meta_version: str = "v1"


class PBModelReader(ModelReader):
    def __init__(self, pb_name, pb_buffer) -> None:
        self.pb_name = pb_name
        self.pb_buffer = pb_buffer

    def read_meta(self) -> ModelMeta:
        return PBModelMeta(dict(pb_name=self.pb_name))

    def read_bytes(self):
        return self.pb_buffer


class PBModelsLoader(ModelsLoader):
    """
    implement of modelloader for fate.components used by fate.flow
    """

    def __init__(self, models) -> None:
        self.models = models

    def names(self) -> typing.List[typing.Tuple[str, str, str]]:
        return list(self.models.keys())

    @contextlib.contextmanager
    def with_name(self, name: typing.Tuple[str, str, str]) -> Iterator[ModelReader]:
        if (pb_pair := self.models.get(name)) is not None:
            model_reader = PBModelReader(*pb_pair)
            yield model_reader

    @classmethod
    def parse(cls, cpn_model_input):
        models = {}
        for model_type, models in cpn_model_input.items():
            for cpn_name, cpn_models in models.items():
                for model_name, (pb_name, pb_buffer) in cpn_models.items():
                    models[(model_type, cpn_name, model_name)] = (
                        pb_name,
                        pb_buffer,
                    )
        return PBModelsLoader(models)

    @property
    def has_model(self):
        return "model" in [model_type for model_type, _, _ in self.models.keys()]

    @property
    def has_isometric_model(self):
        return "isometric_model" in [
            model_type for model_type, _, _ in self.models.keys()
        ]


class PBModelWriter(ModelWriter):
    def __init__(self) -> None:
        self.meta = None
        self.buffer = None

    def write_meta(self, meta: ModelMeta):
        self.meta = meta

    def write_bytes(self, buffer):
        self.buffer = buffer


class PBModelsSaver(ModelsSaver):
    def __init__(self) -> None:
        self.models = {}

    @contextlib.contextmanager
    def with_name(self, name: typing.Tuple[(str, str, str)]) -> Iterator[PBModelWriter]:
        model_writer = PBModelWriter()
        yield model_writer
        self.models[name] = (model_writer.meta, model_writer.buffer)
