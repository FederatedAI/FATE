import json
import logging
import typing
from pathlib import Path
from typing import Dict, Optional, Union

import requests
from fate.components.core.essential import JsonMetricArtifactType

logger = logging.getLogger(__name__)

from .._base_type import (
    URI,
    ArtifactDescribe,
    Metadata,
    MetricOutputMetadata,
    _ArtifactType,
    _ArtifactTypeWriter,
)

if typing.TYPE_CHECKING:
    from fate.arch import Context


class JsonMetricFileWriter(_ArtifactTypeWriter[MetricOutputMetadata]):
    def write(self, data, metadata: Optional[Dict] = None):
        self.artifact.consumed()
        path = Path(self.artifact.uri.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fw:
            json.dump(data, fw)

        if metadata is not None:
            self.artifact.metadata.metadata = metadata


class JsonMetricRestfulWriter(_ArtifactTypeWriter[MetricOutputMetadata]):
    def write(self, data):
        self.artifact.consumed()
        try:
            output = requests.post(url=self.artifact.uri.original_uri, json=dict(data=[data]))
        except Exception as e:
            logger.error(f"write data `{data}` to {self.artifact.uri.original_uri} failed, error: {e}")
        else:
            logger.debug(f"write data `{data}` to {self.artifact.uri.original_uri} success, output: {output}")

    def write_metadata(self, metadata: Dict):
        self.artifact.metadata.metadata = metadata

    def close(self):
        pass


class JsonMetricArtifactDescribe(ArtifactDescribe[JsonMetricArtifactType, MetricOutputMetadata]):
    @classmethod
    def get_type(cls):
        return JsonMetricArtifactType

    def get_writer(
        self, config, ctx: "Context", uri: URI, type_name: str
    ) -> Union[JsonMetricFileWriter, JsonMetricRestfulWriter]:
        if uri.scheme == "http" or uri.scheme == "https":
            return JsonMetricRestfulWriter(
                ctx, _ArtifactType(uri=uri, metadata=MetricOutputMetadata(), type_name=type_name)
            )
        elif uri.scheme == "file":
            return JsonMetricFileWriter(
                ctx, _ArtifactType(uri=uri, metadata=MetricOutputMetadata(), type_name=type_name)
            )
        else:
            raise ValueError(f"unsupported uri scheme: {uri.scheme}")

    def get_reader(self, ctx: "Context", uri: URI, metadata: Metadata, type_name: str):
        raise NotImplementedError()
