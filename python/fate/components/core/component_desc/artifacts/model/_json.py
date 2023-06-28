import json
import typing
from pathlib import Path

from fate.components.core.essential import JsonModelArtifactType

from .._base_type import (
    URI,
    ArtifactDescribe,
    Metadata,
    ModelOutputMetadata,
    _ArtifactType,
    _ArtifactTypeReader,
    _ArtifactTypeWriter,
)

if typing.TYPE_CHECKING:
    from fate.arch import Context


class JsonModelWriter(_ArtifactTypeWriter[ModelOutputMetadata]):
    def write(self, data, metadata: dict = None):
        path = Path(self.artifact.uri.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fw:
            json.dump(data, fw)
        if metadata is not None:
            self.artifact.metadata.metadata = metadata


class JsonModelReader(_ArtifactTypeReader):
    def read(self):
        try:
            with open(self.artifact.uri.path, "r") as fr:
                return json.load(fr)
        except Exception as e:
            raise RuntimeError(f"load json model named from {self.artifact} failed: {e}")


class JsonModelArtifactDescribe(ArtifactDescribe[JsonModelArtifactType, ModelOutputMetadata]):
    @classmethod
    def get_type(cls):
        return JsonModelArtifactType

    def get_writer(self, ctx: "Context", uri: URI, type_name: str) -> JsonModelWriter:
        return JsonModelWriter(ctx, _ArtifactType(uri=uri, metadata=ModelOutputMetadata(), type_name=type_name))

    def get_reader(self, ctx: "Context", uri: URI, metadata: Metadata, type_name: str) -> JsonModelReader:
        return JsonModelReader(ctx, _ArtifactType(uri=uri, metadata=metadata, type_name=type_name))
