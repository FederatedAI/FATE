import typing
from pathlib import Path

from fate.components.core.essential import DataDirectoryArtifactType

from .._base_type import ArtifactDescribe, Metadata, _ArtifactType, _ArtifactTypeWriter

if typing.TYPE_CHECKING:
    from fate.arch import URI


class DataDirectoryWriter(_ArtifactTypeWriter):
    def get_directory(self) -> Path:
        path = Path(self.artifact.uri.path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_metadata(self, metadata: dict, name=None, namespace=None):
        self.artifact.metadata.metadata.update(metadata)
        if name is not None:
            self.artifact.metadata.name = name
        if namespace is not None:
            self.artifact.metadata.namespace = namespace


class DataDirectoryArtifactDescribe(ArtifactDescribe):
    def get_type(self):
        return DataDirectoryArtifactType

    def get_writer(self, uri: "URI", metadata: Metadata) -> _ArtifactTypeWriter:
        return DataDirectoryWriter(_ArtifactType(uri, metadata))

    def _load_as_component_execute_arg(self, ctx, artifact: _ArtifactType):
        return artifact
