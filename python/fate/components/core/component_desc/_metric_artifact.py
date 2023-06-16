import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional

from .._role import Role
from ._artifact_base import (
    URI,
    ArtifactDescribe,
    ArtifactType,
    ComponentArtifactDescribes,
    Metadata,
)


class MetricArtifactType(ArtifactType):
    type = "metric"


class JsonMetricArtifactType(MetricArtifactType):
    type = "metrics_json"

    def __init__(self, path, metadata: Metadata) -> None:
        self.path = path
        self.metadata = metadata

    @classmethod
    def _load(cls, uri: URI, metadata: Metadata):
        return cls(uri.path, metadata)

    def dict(self):
        return {"metadata": self.metadata, "uri": f"file://{self.path}"}


class JsonMetricWriter:
    def __init__(self, artifact: JsonMetricArtifactType) -> None:
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


class JsonMetricArtifactDescribe(ArtifactDescribe[JsonMetricArtifactType]):
    def _get_type(self):
        return JsonMetricArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: JsonMetricArtifactType):
        try:
            with open(artifact.path, "r") as fr:
                return json.load(fr)
        except Exception as e:
            raise RuntimeError(f"load json model named from {artifact} failed: {e}")

    def _load_as_component_execute_arg_writer(self, ctx, artifact: JsonMetricArtifactType):
        return JsonMetricWriter(artifact)


def json_metric_output(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _output_metric_artifact(_create_json_metric_artifact_describe(name, roles, desc, optional, multi=False))


def json_metric_outputs(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _output_metric_artifact(_create_json_metric_artifact_describe(name, roles, desc, optional, multi=True))


def _output_metric_artifact(desc):
    def decorator(f):
        if not hasattr(f, "__component_artifacts__"):
            f.__component_artifacts__ = ComponentArtifactDescribes()

        f.__component_artifacts__.add_metric_output(desc)
        return f

    return decorator


def _prepare(roles, desc):
    if roles is None:
        roles = []
    else:
        roles = [r.name for r in roles]
    if desc:
        desc = inspect.cleandoc(desc)
    return roles, desc


def _create_json_metric_artifact_describe(name, roles, desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return JsonMetricArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)
