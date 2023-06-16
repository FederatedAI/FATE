from pathlib import Path

from .._base_type import URI, ArtifactDescribe, ArtifactType, Metadata


class DataDirectoryArtifactType(ArtifactType):
    type = "data_directory"

    def __init__(self, path, metadata: Metadata) -> None:
        self.path = path
        self.metadata = metadata

    @classmethod
    def _load(cls, uri: URI, metadata: Metadata):
        return DataDirectoryArtifactType(uri.path, metadata)

    def dict(self):
        return {
            "metadata": self.metadata,
            "uri": f"file://{self.path}",
        }


class DataDirectoryWriter:
    def __init__(self, artifact: DataDirectoryArtifactType) -> None:
        self.artifact = artifact

    def get_directory(self) -> Path:
        path = Path(self.artifact.path)
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
        return str(self)


class DataDirectoryArtifactDescribe(ArtifactDescribe):
    def _get_type(self):
        return DataDirectoryArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: DataDirectoryArtifactType):
        return artifact

    def _load_as_component_execute_arg_writer(self, ctx, artifact: DataDirectoryArtifactType):
        return DataDirectoryWriter(artifact)
