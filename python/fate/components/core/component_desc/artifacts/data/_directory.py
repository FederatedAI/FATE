from pathlib import Path

from fate.components.core.essential import DataDirectoryArtifactType

from .._base_type import (
    URI,
    ArtifactDescribe,
    DataOutputMetadata,
    _ArtifactType,
    _ArtifactTypeReader,
    _ArtifactTypeWriter,
)


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


class DataDirectoryReader(_ArtifactTypeReader):
    def get_directory(self) -> Path:
        path = Path(self.artifact.uri.path)
        return path

    def get_metadata(self):
        return self.artifact.metadata.metadata


class DataDirectoryArtifactDescribe(ArtifactDescribe):
    def get_type(self):
        return DataDirectoryArtifactType

    def get_writer(self, ctx, uri: URI) -> DataDirectoryWriter:
        return DataDirectoryWriter(ctx, _ArtifactType(uri=uri, metadata=DataOutputMetadata()))

    def get_reader(self, ctx, artifact_type: _ArtifactType) -> DataDirectoryReader:
        return DataDirectoryReader(ctx, artifact_type)
