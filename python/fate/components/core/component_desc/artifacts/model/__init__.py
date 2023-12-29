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

from typing import Iterator, List, Optional, Type

from ._directory import (
    ModelDirectoryArtifactDescribe,
    ModelDirectoryReader,
    ModelDirectoryWriter,
)
from ._json import JsonModelArtifactDescribe, JsonModelReader, JsonModelWriter
from ._unresolved import ModelUnresolvedArtifactDescribe, ModelUnresolvedReader, ModelUnresolvedWriter
from .._base_type import Role, _create_artifact_annotation


def json_model_input(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[JsonModelReader]:
    return _create_artifact_annotation(True, False, JsonModelArtifactDescribe, "model")(roles, desc, optional)


def json_model_inputs(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[List[JsonModelReader]]:
    return _create_artifact_annotation(True, True, JsonModelArtifactDescribe, "model")(roles, desc, optional)


def json_model_output(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[JsonModelWriter]:
    return _create_artifact_annotation(False, False, JsonModelArtifactDescribe, "model")(roles, desc, optional)


def json_model_outputs(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[Iterator[JsonModelWriter]]:
    return _create_artifact_annotation(False, True, JsonModelArtifactDescribe, "model")(roles, desc, optional)


def model_directory_input(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[ModelDirectoryReader]:
    return _create_artifact_annotation(True, False, ModelDirectoryArtifactDescribe, "model")(roles, desc, optional)


def model_directory_inputs(
    roles: Optional[List[Role]] = None, desc="", optional=False
) -> Type[List[ModelDirectoryReader]]:
    return _create_artifact_annotation(True, True, ModelDirectoryArtifactDescribe, "model")(roles, desc, optional)


def model_directory_output(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[ModelDirectoryWriter]:
    return _create_artifact_annotation(False, False, ModelDirectoryArtifactDescribe, "model")(roles, desc, optional)


def model_directory_outputs(
    roles: Optional[List[Role]] = None, desc="", optional=False
) -> Type[Iterator[ModelDirectoryWriter]]:
    return _create_artifact_annotation(False, True, ModelDirectoryArtifactDescribe, "model")(roles, desc, optional)


def model_unresolved_output(
    roles: Optional[List[Role]] = None, desc="", optional=False
) -> Type[ModelUnresolvedWriter]:
    return _create_artifact_annotation(False, False, ModelUnresolvedArtifactDescribe, "model")(roles, desc, optional)


def model_unresolved_outputs(
    roles: Optional[List[Role]] = None, desc="", optional=False
) -> Type[Iterator[ModelUnresolvedWriter]]:
    return _create_artifact_annotation(False, True, ModelUnresolvedArtifactDescribe, "model")(roles, desc, optional)
