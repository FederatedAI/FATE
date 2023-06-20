import json
import typing
from pathlib import Path
from typing import Dict, Optional

from fate.components.core.essential import JsonMetricArtifactType

from .._base_type import ArtifactDescribe, Metadata, _ArtifactType, _ArtifactTypeWriter

if typing.TYPE_CHECKING:
    from fate.arch import URI


class JsonMetricWriter(_ArtifactTypeWriter):
    def write(self, data, metadata: Optional[Dict] = None):
        path = Path(self.artifact.uri.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fw:
            json.dump(data, fw)

        if metadata is not None:
            self.artifact.metadata.metadata = metadata


class JsonMetricArtifactDescribe(ArtifactDescribe[_ArtifactType]):
    def get_type(self):
        return JsonMetricArtifactType

    def get_writer(self, uri: "URI", metadata: Metadata) -> _ArtifactTypeWriter:
        return JsonMetricWriter(_ArtifactType(uri, metadata))

    def _load_as_component_execute_arg(self, ctx, artifact: _ArtifactType):
        try:
            with open(artifact.uri.path, "r") as fr:
                return json.load(fr)
        except Exception as e:
            raise RuntimeError(f"load json model named from {artifact} failed: {e}")
