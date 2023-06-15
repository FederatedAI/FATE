import inspect
import json
from pathlib import Path
from typing import List, Optional

from .._role import Role
from ._artifact_base import (
    URI,
    ArtifactDescribe,
    ArtifactType,
    ComponentArtifactDescribes,
)


class ModelArtifactType(ArtifactType):
    type = "model"


class JsonModelArtifactType(ModelArtifactType):
    type = "model_json"

    def __init__(self, path, metadata) -> None:
        self.path = path
        self.metadata = metadata

    @classmethod
    def _load(cls, uri: URI, metadata):
        return cls(uri.path, metadata)

    def dict(self):
        return {"metadata": self.metadata, "uri": f"file://{self.path}"}


class ModelDirectoryArtifactType(ModelArtifactType):
    type = "model_directory"

    def __init__(self, path, metadata) -> None:
        self.path = path
        self.metadata = metadata

    @classmethod
    def _load(cls, uri: URI, metadata):
        return cls(uri.path, metadata)

    def dict(self):
        return {"metadata": self.metadata, "uri": f"file://{self.path}"}


class JsonModelWriter:
    def __init__(self, artifact: JsonModelArtifactType) -> None:
        self._artifact = artifact

    def write(self, data):
        path = Path(self._artifact.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fw:
            json.dump(data, fw)


class ModelDirectoryWriter:
    def __init__(self, artifact: ModelDirectoryArtifactType) -> None:
        self._artifact = artifact

    def write(self, data):
        path = Path(self._artifact.path)
        path.mkdir(parents=True, exist_ok=True)
        return self._artifact.path


class JsonModelArtifactDescribe(ArtifactDescribe[JsonModelArtifactType]):
    def _get_type(self):
        return JsonModelArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: JsonModelArtifactType):
        try:
            with open(artifact.path, "r") as fr:
                return json.load(fr)
        except Exception as e:
            raise RuntimeError(f"load json model from {artifact} failed: {e}")

    def _load_as_component_execute_arg_writer(self, ctx, artifact: JsonModelArtifactType):
        return JsonModelWriter(artifact)


class ModelDirectoryArtifactDescribe(ArtifactDescribe[ModelDirectoryArtifactType]):
    def _get_type(self):
        return ModelDirectoryArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: ModelDirectoryArtifactType):
        return artifact

    def _load_as_component_execute_arg_writer(self, ctx, artifact: ModelDirectoryArtifactType):
        return ModelDirectoryWriter(artifact)


def json_model_input(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _input_model_artifact(_create_json_model_artifact_describe(name, roles, desc, optional, multi=False))


def json_model_inputs(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _input_model_artifact(_create_json_model_artifact_describe(name, roles, desc, optional, multi=True))


def model_directory_input(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _input_model_artifact(_create_model_directory_artifact_describe(name, roles, desc, optional, multi=False))


def model_directory_inputs(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _input_model_artifact(_create_model_directory_artifact_describe(name, roles, desc, optional, multi=True))


def json_model_output(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _output_model_artifact(_create_json_model_artifact_describe(name, roles, desc, optional, multi=False))


def json_model_outputs(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _output_model_artifact(_create_json_model_artifact_describe(name, roles, desc, optional, multi=True))


def model_directory_output(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _output_model_artifact(_create_model_directory_artifact_describe(name, roles, desc, optional, multi=False))


def model_directory_outputs(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _output_model_artifact(_create_model_directory_artifact_describe(name, roles, desc, optional, multi=True))


def _input_model_artifact(desc):
    def decorator(f):
        if not hasattr(f, "__component_artifacts__"):
            f.__component_artifacts__ = ComponentArtifactDescribes()

        f.__component_artifacts__.add_model_input(desc)
        return f

    return decorator


def _output_model_artifact(desc):
    def decorator(f):
        if not hasattr(f, "__component_artifacts__"):
            f.__component_artifacts__ = ComponentArtifactDescribes()

        f.__component_artifacts__.add_model_output(desc)
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


def _create_json_model_artifact_describe(name, roles, desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return JsonModelArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)


def _create_model_directory_artifact_describe(name, roles, desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return ModelDirectoryArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)
