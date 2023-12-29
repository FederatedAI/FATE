#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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


class DataUnresolvedArtifactType(ArtifactType):
    type_name = "data_unresolved"
    path_type = "unresolved"
    uri_types = ["unresolved"]


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


class ModelUnresolvedArtifactType(ArtifactType):
    type_name = "model_unresolved"
    path_type = "unresolved"
    uri_types = ["unresolved"]
