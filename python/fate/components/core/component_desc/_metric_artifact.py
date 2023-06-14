import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional

from .._role import Role
from ._artifact_base import (
    ArtifactDescribe,
    ArtifactType,
    ComponentArtifactDescribes,
    Slot,
    Slots,
)


class MetricArtifactType(ArtifactType):
    type = "metric"


class JsonMetricArtifactType(MetricArtifactType):
    type = "metrics_json"

    def __init__(
        self,
        name: Optional[str] = None,
        uri: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(uri=uri, name=name, metadata=metadata)


class JsonMetricWriter:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, data):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as fw:
            json.dump(data, fw)


class JsonMetricWriterGenerator:
    def __init__(self, path: Path) -> None:
        self._path = path

    def get_writer(self, index) -> JsonMetricWriter:
        ...


class JsonMetricArtifactDescribe(ArtifactDescribe):
    def _get_type(self):
        return JsonMetricArtifactType.type

    def _load_as_input(self, ctx, apply_config):
        def _load_json_model(name, path, metadata):
            try:
                with open(path, "r") as fr:
                    return json.load(fr)
            except Exception as e:
                raise RuntimeError(f"load json model named {name} failed: {e}")

        if self.multi:
            return [_load_json_model(c.name, c.uri, c.metadata) for c in apply_config]
        else:
            return _load_json_model(apply_config.name, apply_config.uri, apply_config.metadata)

    def _load_as_output_slot(self, ctx, apply_config):
        if self.multi:
            return Slots(JsonMetricWriterGenerator(apply_config))
        else:
            return Slot(JsonMetricWriter(apply_config))


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
