import json
import typing
from pathlib import Path

from fate.components.core.essential import JsonModelArtifactType

from .._base_type import (
    ArtifactDescribe,
    _ArtifactType,
    _ArtifactTypeReader,
    _ArtifactTypeWriter,
)

if typing.TYPE_CHECKING:
    from fate.arch import Context


class JsonModelWriter(_ArtifactTypeWriter):
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


class JsonModelArtifactDescribe(ArtifactDescribe[_ArtifactType]):
    def get_type(self):
        return JsonModelArtifactType

    def get_writer(self, ctx: "Context", artifact_type: _ArtifactType) -> JsonModelWriter:
        return JsonModelWriter(ctx, artifact_type)

    def get_reader(self, ctx: "Context", artifact_type: _ArtifactType) -> JsonModelReader:
        return JsonModelReader(ctx, artifact_type)
