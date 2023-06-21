import typing
from pathlib import Path

from fate.components.core.essential import ModelDirectoryArtifactType

from .._base_type import (
    URI,
    ArtifactDescribe,
    ModelOutputMetadata,
    _ArtifactType,
    _ArtifactTypeReader,
    _ArtifactTypeWriter,
)

if typing.TYPE_CHECKING:
    from fate.arch import Context


class ModelDirectoryWriter(_ArtifactTypeWriter):
    def get_directory(self):
        path = Path(self.artifact.uri.path)
        path.mkdir(parents=True, exist_ok=True)
        return self.artifact.uri.path

    def write_metadata(self, metadata: dict):
        self.artifact.metadata.metadata = metadata


class ModelDirectoryReader(_ArtifactTypeReader):
    def get_directory(self):
        path = Path(self.artifact.uri.path)
        return path

    def get_metadata(self):
        return self.artifact.metadata.metadata


class ModelDirectoryArtifactDescribe(ArtifactDescribe[_ArtifactType]):
    def get_type(self):
        return ModelDirectoryArtifactType

    def get_writer(self, ctx: "Context", uri: URI) -> ModelDirectoryWriter:
        return ModelDirectoryWriter(ctx, _ArtifactType(uri=uri, metadata=ModelOutputMetadata()))

    def get_reader(self, ctx: "Context", artifact_type: _ArtifactType) -> ModelDirectoryReader:
        return ModelDirectoryReader(ctx, artifact_type)
