from pathlib import Path

from fate.components.core.essential import DataDirectoryArtifactType

from .._base_type import (
    URI,
    ArtifactDescribe,
    DataOutputMetadata,
    Metadata,
    _ArtifactType,
    _ArtifactTypeReader,
    _ArtifactTypeWriter,
)


class DataDirectoryWriter(_ArtifactTypeWriter[DataDirectoryArtifactType]):
    def get_directory(self) -> Path:
        self.artifact.consumed()
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
        self.artifact.consumed()
        path = Path(self.artifact.uri.path)
        return path

    def get_metadata(self):
        return self.artifact.metadata.metadata


class DataDirectoryArtifactDescribe(ArtifactDescribe[DataDirectoryArtifactType, DataOutputMetadata]):
    @classmethod
    def get_type(cls):
        return DataDirectoryArtifactType

    def get_writer(self, config, ctx, uri: URI, type_name: str) -> DataDirectoryWriter:
        return DataDirectoryWriter(ctx, _ArtifactType(uri=uri, metadata=DataOutputMetadata(), type_name=type_name))

    def get_reader(self, ctx, uri: "URI", metadata: "Metadata", type_name: str) -> DataDirectoryReader:
        return DataDirectoryReader(ctx, _ArtifactType(uri=uri, metadata=metadata, type_name=type_name))
