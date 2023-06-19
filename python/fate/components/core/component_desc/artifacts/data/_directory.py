import typing
from pathlib import Path

from fate.components.core.essential import DataDirectoryArtifactType

from .._base_type import ArtifactDescribe, _ArtifactType


class _DataDirectoryArtifactType(_ArtifactType["DataDirectoryWriter"]):
    type = DataDirectoryArtifactType

    def get_writer(self) -> "DataDirectoryWriter":
        return DataDirectoryWriter(self)


class DataDirectoryWriter:
    def __init__(self, artifact: _DataDirectoryArtifactType) -> None:
        self.artifact = artifact

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

    def __str__(self):
        return f"DataDirectoryWriter({self.artifact})"

    def __repr__(self):
        return self.__str__()


class DataDirectoryArtifactDescribe(ArtifactDescribe):
    def get_type(self):
        return _DataDirectoryArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: _DataDirectoryArtifactType):
        return artifact
