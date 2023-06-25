from .._base_type import _create_artifact_annotation
from ._dataframe import DataframeArtifactDescribe
from ._directory import DataDirectoryArtifactDescribe
from ._table import TableArtifactDescribe

dataframe_input = _create_artifact_annotation(True, False, DataframeArtifactDescribe, "data")
dataframe_inputs = _create_artifact_annotation(True, True, DataframeArtifactDescribe, "data")
dataframe_output = _create_artifact_annotation(False, False, DataframeArtifactDescribe, "data")
dataframe_outputs = _create_artifact_annotation(False, True, DataframeArtifactDescribe, "data")
table_input = _create_artifact_annotation(True, False, TableArtifactDescribe, "data")
table_inputs = _create_artifact_annotation(True, True, TableArtifactDescribe, "data")
table_output = _create_artifact_annotation(False, False, TableArtifactDescribe, "data")
table_outputs = _create_artifact_annotation(False, True, TableArtifactDescribe, "data")
data_directory_input = _create_artifact_annotation(True, False, DataDirectoryArtifactDescribe, "data")
data_directory_inputs = _create_artifact_annotation(True, True, DataDirectoryArtifactDescribe, "data")
data_directory_output = _create_artifact_annotation(False, False, DataDirectoryArtifactDescribe, "data")
data_directory_outputs = _create_artifact_annotation(False, True, DataDirectoryArtifactDescribe, "data")
