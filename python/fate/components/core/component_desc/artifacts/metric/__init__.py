from .._base_type import _create_artifact_annotation
from ._json import JsonMetricArtifactDescribe

json_metric_output = _create_artifact_annotation(False, False, JsonMetricArtifactDescribe, "metric")
json_metric_outputs = _create_artifact_annotation(False, True, JsonMetricArtifactDescribe, "metric")
