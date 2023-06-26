import json
import typing
from pathlib import Path
from typing import Dict, Optional

from fate.components.core.essential import JsonMetricArtifactType

from .._base_type import (
    URI,
    ArtifactDescribe,
    Metadata,
    MetricOutputMetadata,
    _ArtifactType,
    _ArtifactTypeReader,
    _ArtifactTypeWriter,
)

if typing.TYPE_CHECKING:
    from fate.arch import Context


class JsonMetricWriter(_ArtifactTypeWriter[MetricOutputMetadata]):
    def write(self, data, metadata: Optional[Dict] = None):
        path = Path(self.artifact.uri.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fw:
            json.dump(data, fw)

        if metadata is not None:
            self.artifact.metadata.metadata = metadata


class JsonMetricReader(_ArtifactTypeReader):
    def read(self):
        try:
            with open(self.artifact.uri.path, "r") as fr:
                return json.load(fr)
        except Exception as e:
            raise RuntimeError(f"load json model named from {self.artifact} failed: {e}")


class JsonMetricArtifactDescribe(ArtifactDescribe[JsonMetricArtifactType, MetricOutputMetadata]):
    @classmethod
    def get_type(cls):
        return JsonMetricArtifactType

    def get_writer(self, ctx: "Context", uri: URI) -> JsonMetricWriter:
        return JsonMetricWriter(ctx, _ArtifactType(uri=uri, metadata=MetricOutputMetadata()))

    def get_reader(self, ctx: "Context", uri: URI, metadata: Metadata) -> JsonMetricReader:
        return JsonMetricReader(ctx, _ArtifactType(uri=uri, metadata=metadata))
