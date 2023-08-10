import datetime
import typing
from pathlib import Path

from fate.components.core.essential import ModelDirectoryArtifactType

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


class ModelDirectoryWriter(_ArtifactTypeWriter[ModelOutputMetadata]):
    def get_directory(self):
        self.artifact.consumed()
        path = Path(self.artifact.uri.path)
        path.mkdir(parents=True, exist_ok=True)

        # update model overview
        from fate.components.core.spec.model import MLModelModelSpec

        model_overview = self.artifact.metadata.model_overview
        model_overview.party.models.append(
            MLModelModelSpec(
                name="",
                created_time=datetime.datetime.now().isoformat(),
                file_format=ModelDirectoryArtifactType.type_name,
                metadata={},
            )
        )
        return self.artifact.uri.path

    def write_metadata(self, metadata: dict):
        self.artifact.metadata.metadata = metadata


class ModelDirectoryReader(_ArtifactTypeReader):
    def get_directory(self):
        self.artifact.consumed()
        path = Path(self.artifact.uri.path)
        return path

    def get_metadata(self):
        return self.artifact.metadata.metadata


class ModelDirectoryArtifactDescribe(ArtifactDescribe[ModelDirectoryArtifactType, ModelOutputMetadata]):
    @classmethod
    def get_type(cls):
        return ModelDirectoryArtifactType

    def get_writer(self, config, ctx: "Context", uri: URI, type_name: str) -> ModelDirectoryWriter:
        return ModelDirectoryWriter(ctx, _ArtifactType(uri=uri, metadata=ModelOutputMetadata(), type_name=type_name))

    def get_reader(self, ctx: "Context", uri: URI, metadata: Metadata, type_name: str) -> ModelDirectoryReader:
        return ModelDirectoryReader(ctx, _ArtifactType(uri=uri, metadata=metadata, type_name=type_name))
