import json
from pathlib import Path

from fate.components.core.essential import JsonModelArtifactType

from .._base_type import ArtifactDescribe, _ArtifactType


class _JsonModelArtifactType(_ArtifactType["JsonModelWriter"]):
    type = JsonModelArtifactType

    def get_writer(self) -> "JsonModelWriter":
        return JsonModelWriter(self)


class JsonModelWriter:
    def __init__(self, artifact: _JsonModelArtifactType) -> None:
        self._artifact = artifact

    def write(self, data):
        path = Path(self._artifact.uri.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fw:
            json.dump(data, fw)

    def __str__(self):
        return f"JsonModelWriter({self._artifact})"


class JsonModelArtifactDescribe(ArtifactDescribe[_JsonModelArtifactType]):
    def get_type(self):
        return _JsonModelArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: _JsonModelArtifactType):
        try:
            with open(artifact.uri.path, "r") as fr:
                return json.load(fr)
        except Exception as e:
            raise RuntimeError(f"load json model from {artifact} failed: {e}")
