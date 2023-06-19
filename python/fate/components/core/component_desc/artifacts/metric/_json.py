import json
from pathlib import Path
from typing import Dict, Optional

from fate.components.core.essential import JsonMetricArtifactType

from .._base_type import URI, ArtifactDescribe, Metadata, _ArtifactType


class _JsonMetricArtifactType(_ArtifactType):
    type = JsonMetricArtifactType

    def __init__(self, path, metadata: Metadata) -> None:
        self.path = path
        self.metadata = metadata

    @classmethod
    def _load(cls, uri: URI, metadata: Metadata):
        return cls(uri.path, metadata)

    def dict(self):
        return {"metadata": self.metadata, "uri": f"file://{self.path}"}


class JsonMetricWriter:
    def __init__(self, artifact: _JsonMetricArtifactType) -> None:
        self._artifact = artifact

    def write(
        self, data, metadata: Optional[Dict] = None, namespace: Optional[str] = None, name: Optional[str] = None
    ):
        if metadata is not None:
            self._artifact.metadata.metadata.update(metadata)
        if namespace is not None:
            self._artifact.metadata.namespace = namespace
        if name is not None:
            self._artifact.metadata.name = name

        path = Path(self._artifact.path)
        path.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fw:
            json.dump(data, fw)


class JsonMetricArtifactDescribe(ArtifactDescribe[_JsonMetricArtifactType]):
    def get_type(self):
        return _JsonMetricArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: _JsonMetricArtifactType):
        try:
            with open(artifact.path, "r") as fr:
                return json.load(fr)
        except Exception as e:
            raise RuntimeError(f"load json model named from {artifact} failed: {e}")

    def _load_as_component_execute_arg_writer(self, ctx, artifact: _JsonMetricArtifactType):
        return JsonMetricWriter(artifact)
