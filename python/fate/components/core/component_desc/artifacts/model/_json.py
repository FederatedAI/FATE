import json
import typing
from pathlib import Path

from fate.components.core.essential import JsonModelArtifactType

from .._base_type import ArtifactDescribe, Metadata, _ArtifactType, _ArtifactTypeWriter

if typing.TYPE_CHECKING:
    from fate.arch import URI


class JsonModelWriter(_ArtifactTypeWriter):
    def write(self, data):
        path = Path(self.artifact.uri.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fw:
            json.dump(data, fw)


class JsonModelArtifactDescribe(ArtifactDescribe[_ArtifactType]):
    def get_type(self):
        return JsonModelArtifactType

    def get_writer(self, uri: "URI", metadata: Metadata) -> _ArtifactTypeWriter:
        return JsonModelWriter(_ArtifactType(uri, metadata))

    def _load_as_component_execute_arg(self, ctx, artifact: _ArtifactType):
        try:
            with open(artifact.uri.path, "r") as fr:
                return json.load(fr)
        except Exception as e:
            raise RuntimeError(f"load json model from {artifact} failed: {e}")
