from typing import Iterator, List, Optional, Type

from .._base_type import Role, _create_artifact_annotation
from ._directory import (
    ModelDirectoryArtifactDescribe,
    ModelDirectoryReader,
    ModelDirectoryWriter,
)
from ._json import JsonModelArtifactDescribe, JsonModelReader, JsonModelWriter


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
