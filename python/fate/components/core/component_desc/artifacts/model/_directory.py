import typing
from pathlib import Path

from fate.components.core.essential import ModelDirectoryArtifactType

from .._base_type import ArtifactDescribe, _ArtifactType


class _ModelDirectoryArtifactType(_ArtifactType["ModelDirectoryWriter"]):
    type = ModelDirectoryArtifactType

    def get_writer(self) -> "ModelDirectoryWriter":
        return ModelDirectoryWriter(self)


class ModelDirectoryWriter:
    def __init__(self, artifact: _ModelDirectoryArtifactType) -> None:
        self._artifact = artifact

    def write(self, data):
        path = Path(self._artifact.uri.path)
        path.mkdir(parents=True, exist_ok=True)
        return self._artifact.uri.path

    def __str__(self):
        return f"ModelDirectoryWriter({self._artifact})"


class ModelDirectoryArtifactDescribe(ArtifactDescribe[_ModelDirectoryArtifactType]):
    def get_type(self):
        return _ModelDirectoryArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: _ModelDirectoryArtifactType):
        return artifact
