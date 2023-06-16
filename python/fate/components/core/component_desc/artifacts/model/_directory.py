from pathlib import Path

from .._base_type import URI, ArtifactDescribe, ArtifactType, Metadata


class ModelDirectoryArtifactType(ArtifactType):
    type = "model_directory"

    def __init__(self, path, metadata: Metadata) -> None:
        self.path = path
        self.metadata = metadata

    @classmethod
    def _load(cls, uri: URI, metadata: Metadata):
        return cls(uri.path, metadata)

    def dict(self):
        return {"metadata": self.metadata, "uri": f"file://{self.path}"}


class ModelDirectoryWriter:
    def __init__(self, artifact: ModelDirectoryArtifactType) -> None:
        self._artifact = artifact

    def write(self, data):
        path = Path(self._artifact.path)
        path.mkdir(parents=True, exist_ok=True)
        return self._artifact.path


class ModelDirectoryArtifactDescribe(ArtifactDescribe[ModelDirectoryArtifactType]):
    def _get_type(self):
        return ModelDirectoryArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: ModelDirectoryArtifactType):
        return artifact

    def _load_as_component_execute_arg_writer(self, ctx, artifact: ModelDirectoryArtifactType):
        return ModelDirectoryWriter(artifact)
