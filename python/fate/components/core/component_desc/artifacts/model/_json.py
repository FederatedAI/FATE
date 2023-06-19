import json
from pathlib import Path

from fate.components.core.essential import JsonModelArtifactType

from .._base_type import URI, ArtifactDescribe, Metadata, _ArtifactType


class _JsonModelArtifactType(_ArtifactType):
    type = JsonModelArtifactType

    def __init__(self, path, metadata: Metadata) -> None:
        self.path = path
        self.metadata = metadata

    @classmethod
    def _load(cls, uri: URI, metadata: Metadata):
        return cls(uri.path, metadata)

    def dict(self):
        return {"metadata": self.metadata, "uri": f"file://{self.path}"}


class JsonModelWriter:
    def __init__(self, artifact: _JsonModelArtifactType) -> None:
        self._artifact = artifact

    def write(self, data):
        path = Path(self._artifact.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fw:
            json.dump(data, fw)


class JsonModelArtifactDescribe(ArtifactDescribe[_JsonModelArtifactType]):
    def get_type(self):
        return _JsonModelArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: _JsonModelArtifactType):
        try:
            with open(artifact.path, "r") as fr:
                return json.load(fr)
        except Exception as e:
            raise RuntimeError(f"load json model from {artifact} failed: {e}")

    def _load_as_component_execute_arg_writer(self, ctx, artifact: _JsonModelArtifactType):
        return JsonModelWriter(artifact)
