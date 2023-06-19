import json
import typing
from pathlib import Path
from typing import Dict, Optional

from fate.components.core.essential import JsonMetricArtifactType

from .._base_type import ArtifactDescribe, _ArtifactType


class _JsonMetricArtifactType(_ArtifactType["JsonMetricWriter"]):
    type = JsonMetricArtifactType

    def get_writer(self) -> "JsonMetricWriter":
        return JsonMetricWriter(self)


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

    def __str__(self):
        return f"JsonMetricWriter({self._artifact})"


class JsonMetricArtifactDescribe(ArtifactDescribe[_JsonMetricArtifactType]):
    def get_type(self):
        return _JsonMetricArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: _JsonMetricArtifactType):
        try:
            with open(artifact.uri.path, "r") as fr:
                return json.load(fr)
        except Exception as e:
            raise RuntimeError(f"load json model named from {artifact} failed: {e}")
