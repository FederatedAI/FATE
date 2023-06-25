from .._base_type import _create_artifact_annotation
from ._directory import ModelDirectoryArtifactDescribe
from ._json import JsonModelArtifactDescribe

json_model_input = _create_artifact_annotation(True, False, JsonModelArtifactDescribe, "model")
json_model_inputs = _create_artifact_annotation(True, True, JsonModelArtifactDescribe, "model")
json_model_output = _create_artifact_annotation(False, False, JsonModelArtifactDescribe, "model")
json_model_outputs = _create_artifact_annotation(False, True, JsonModelArtifactDescribe, "model")
model_directory_input = _create_artifact_annotation(True, False, ModelDirectoryArtifactDescribe, "model")
model_directory_inputs = _create_artifact_annotation(True, True, ModelDirectoryArtifactDescribe, "model")
model_directory_output = _create_artifact_annotation(False, False, ModelDirectoryArtifactDescribe, "model")
model_directory_outputs = _create_artifact_annotation(False, True, ModelDirectoryArtifactDescribe, "model")
