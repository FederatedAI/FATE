import typing
from typing import Union

from fate.components.core.essential import TableArtifactType

from .._base_type import ArtifactDescribe, _ArtifactType


class _TableArtifactType(_ArtifactType["TableWriter"]):
    type = TableArtifactType

    def get_writer(self) -> "TableWriter":
        return TableWriter(self)


class TableWriter:
    def __init__(self, artifact: _TableArtifactType) -> None:
        self.artifact = artifact

    def write(self, table):
        if "schema" not in self.artifact.metadata.metadata:
            self.artifact.metadata.metadata["schema"] = {}
        table.save(
            uri=self.artifact.uri,
            schema=self.artifact.metadata.metadata["schema"],
            options=self.artifact.metadata.metadata.get("options", None),
        )

    def __str__(self):
        return f"TableWriter({self.artifact})"

    def __repr__(self):
        return self.__str__()


class TableArtifactDescribe(ArtifactDescribe):
    def get_type(self):
        return _TableArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: _TableArtifactType):
        return ctx.computing.load(
            uri=artifact.uri,
            schema=artifact.metadata.metadata.get("schema", {}),
            options=artifact.metadata.metadata.get("options", None),
        )
