from pathlib import Path

from fate.components.core.essential import DataUnresolvedArtifactType
from .._base_type import (
    URI,
    ArtifactDescribe,
    DataOutputMetadata,
    Metadata,
    _ArtifactType,
    _ArtifactTypeReader,
    _ArtifactTypeWriter,
)


class DataUnresolvedWriter(_ArtifactTypeWriter[DataUnresolvedArtifactType]):
    def write_metadata(self, metadata: dict, name=None, namespace=None):
        self.artifact.consumed()
        self.artifact.metadata.metadata.update(metadata)
        if name is not None:
            self.artifact.metadata.name = name
        if namespace is not None:
            self.artifact.metadata.namespace = namespace


class DataUnresolvedReader(_ArtifactTypeReader):
    def get_metadata(self):
        return self.artifact.metadata.metadata


class DataUnresolvedArtifactDescribe(ArtifactDescribe[DataUnresolvedArtifactType, DataOutputMetadata]):
    @classmethod
    def get_type(cls):
        return DataUnresolvedArtifactType

    def get_writer(self, config, ctx, uri: URI, type_name: str) -> DataUnresolvedWriter:
        return DataUnresolvedWriter(ctx, _ArtifactType(uri=uri, metadata=DataOutputMetadata(), type_name=type_name))

    def get_reader(self, ctx, uri: "URI", metadata: "Metadata", type_name: str) -> DataUnresolvedReader:
        return DataUnresolvedReader(ctx, _ArtifactType(uri=uri, metadata=metadata, type_name=type_name))
