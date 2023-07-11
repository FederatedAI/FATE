from typing import List, Optional, Type

from .._base_type import Role, _create_artifact_annotation
from ._json import (
    JsonMetricArtifactDescribe,
    JsonMetricFileWriter,
    JsonMetricRestfulWriter,
)


def json_metric_output(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[JsonMetricFileWriter]:
    return _create_artifact_annotation(False, False, JsonMetricArtifactDescribe, "metric")(roles, desc, optional)


def json_metric_outputs(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[JsonMetricFileWriter]:
    return _create_artifact_annotation(False, True, JsonMetricArtifactDescribe, "metric")(roles, desc, optional)
