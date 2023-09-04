from typing import List


class ArtifactType:
    type_name: str
    path_type: str
    uri_types: List[str]


class DataframeArtifactType(ArtifactType):
    type_name = "dataframe"
    path_type = "distributed"
    uri_types = ["eggroll", "hdfs"]


class TableArtifactType(ArtifactType):
    type_name = "table"
    path_type = "distributed"
    uri_types = ["eggroll", "hdfs"]


class DataDirectoryArtifactType(ArtifactType):
    type_name = "data_directory"
    path_type = "directory"
    uri_types = ["file"]


class ModelDirectoryArtifactType(ArtifactType):
    type_name = "model_directory"
    path_type = "directory"
    uri_types = ["file"]


class JsonModelArtifactType(ArtifactType):
    type_name = "json_model"
    path_type = "file"
    uri_types = ["file"]


class JsonMetricArtifactType(ArtifactType):
    type_name = "json_metric"
    path_type = "file"
    uri_types = ["file"]
