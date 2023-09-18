import typing

from fate.components.core.essential import TableArtifactType

from .._base_type import (
    URI,
    ArtifactDescribe,
    DataOutputMetadata,
    Metadata,
    _ArtifactType,
    _ArtifactTypeReader,
    _ArtifactTypeWriter,
)

if typing.TYPE_CHECKING:
    from fate.arch import Context


class TableWriter(_ArtifactTypeWriter[DataOutputMetadata]):
    def write(self, table):
        self.artifact.consumed()
        if "schema" not in self.artifact.metadata.metadata:
            self.artifact.metadata.metadata["schema"] = {}
        table.save(
            uri=self.artifact.uri,
            schema=self.artifact.metadata.metadata["schema"],
            options=self.artifact.metadata.metadata.get("options", None),
        )


class TableReader(_ArtifactTypeReader):
    def read(self):
        self.artifact.consumed()
        return self.ctx.computing.load(
            uri=self.artifact.uri,
            schema=self.artifact.metadata.metadata.get("schema", {}),
            options=self.artifact.metadata.metadata.get("options", None),
        )


class TableArtifactDescribe(ArtifactDescribe[TableArtifactType, DataOutputMetadata]):
    @classmethod
    def get_type(cls):
        return TableArtifactType

    def get_writer(self, config, ctx: "Context", uri: URI, type_name: str) -> TableWriter:
        return TableWriter(ctx, _ArtifactType(uri, DataOutputMetadata(), type_name))

    def get_reader(self, ctx: "Context", uri: "URI", metadata: "Metadata", type_name: str) -> TableReader:
        return TableReader(ctx, _ArtifactType(uri, metadata, type_name))
