import typing

from fate.components.core.essential import TableArtifactType

from .._base_type import (
    ArtifactDescribe,
    _ArtifactType,
    _ArtifactTypeReader,
    _ArtifactTypeWriter,
)

if typing.TYPE_CHECKING:
    from fate.arch import Context


class TableWriter(_ArtifactTypeWriter):
    def write(self, table):
        if "schema" not in self.artifact.metadata.metadata:
            self.artifact.metadata.metadata["schema"] = {}
        table.save(
            uri=self.artifact.uri,
            schema=self.artifact.metadata.metadata["schema"],
            options=self.artifact.metadata.metadata.get("options", None),
        )


class TableReader(_ArtifactTypeReader):
    def read(self):
        return self.ctx.computing.load(
            uri=self.artifact.uri,
            schema=self.artifact.metadata.metadata.get("schema", {}),
            options=self.artifact.metadata.metadata.get("options", None),
        )


class TableArtifactDescribe(ArtifactDescribe):
    def get_type(self):
        return TableArtifactType

    def get_writer(self, ctx, artifact_type: _ArtifactType) -> TableWriter:
        return TableWriter(ctx, artifact_type)

    def get_reader(self, ctx: "Context", artifact_type: _ArtifactType) -> TableReader:
        return TableReader(ctx, artifact_type)
