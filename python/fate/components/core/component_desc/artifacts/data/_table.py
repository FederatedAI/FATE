import typing

from fate.components.core.essential import TableArtifactType

from .._base_type import ArtifactDescribe, Metadata, _ArtifactType, _ArtifactTypeWriter

if typing.TYPE_CHECKING:
    from fate.arch import URI


class TableWriter(_ArtifactTypeWriter):
    def write(self, table):
        if "schema" not in self.artifact.metadata.metadata:
            self.artifact.metadata.metadata["schema"] = {}
        table.save(
            uri=self.artifact.uri,
            schema=self.artifact.metadata.metadata["schema"],
            options=self.artifact.metadata.metadata.get("options", None),
        )


class TableArtifactDescribe(ArtifactDescribe):
    def get_type(self):
        return TableArtifactType

    def get_writer(self, uri: "URI", metadata: Metadata) -> _ArtifactTypeWriter:
        return TableWriter(_ArtifactType(uri, metadata))

    def _load_as_component_execute_arg(self, ctx, artifact: _ArtifactType):
        return ctx.computing.load(
            uri=artifact.uri,
            schema=artifact.metadata.metadata.get("schema", {}),
            options=artifact.metadata.metadata.get("options", None),
        )
