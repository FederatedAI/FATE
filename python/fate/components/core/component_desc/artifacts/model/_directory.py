import typing
from pathlib import Path

from fate.components.core.essential import ModelDirectoryArtifactType

from .._base_type import ArtifactDescribe, Metadata, _ArtifactType, _ArtifactTypeWriter

if typing.TYPE_CHECKING:
    from fate.arch import URI


class ModelDirectoryWriter(_ArtifactTypeWriter):
    def get_directory(self):
        path = Path(self.artifact.uri.path)
        path.mkdir(parents=True, exist_ok=True)
        return self.artifact.uri.path

    def write_metadata(self, metadata: dict):
        self.artifact.metadata.metadata = metadata


class ModelDirectoryArtifactDescribe(ArtifactDescribe[_ArtifactType]):
    def get_type(self):
        return ModelDirectoryArtifactType

    def get_writer(self, uri: "URI", metadata: Metadata) -> _ArtifactTypeWriter:
        return ModelDirectoryWriter(_ArtifactType(uri, metadata))

    def _load_as_component_execute_arg(self, ctx, artifact: _ArtifactType):
        return artifact
