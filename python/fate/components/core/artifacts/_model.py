import inspect
import json
from pathlib import Path
from typing import List, Optional

from fate.components.core.cpn import Role

from ._artifact import ArtifactDescribe, ArtifactType, ComponentArtifactDescribes
from ._slot import Slot, Slots


class ModelArtifactType(ArtifactType):
    type = "model"


class JsonModelArtifactType(ModelArtifactType):
    type = "model_json"


class ModelDirectoryArtifactType(ModelArtifactType):
    type = "model_directory"


class JsonModelWriter:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, data):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as fw:
            json.dump(data, fw)


class ModelDirectoryWriterGenerator:
    def __init__(self, path: Path) -> None:
        self._path = path

    def get_writer(self, index) -> JsonModelWriter:
        ...


class ModelDirectoryWriter:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, data):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as fw:
            json.dump(data, fw)


class JsonModelWriterGenerator:
    def __init__(self, path: Path) -> None:
        self._path = path

    def get_writer(self, index) -> JsonModelWriter:
        ...


class JsonModelArtifactDescribe(ArtifactDescribe):
    def _get_type(self):
        return JsonModelArtifactType.type

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
            return Slots(JsonModelWriterGenerator(apply_config))
        else:
            return Slot(JsonModelWriter(apply_config))


class ModelDirectoryArtifactDescribe(ArtifactDescribe):
    def _get_type(self):
        return ModelDirectoryArtifactType.type

    def _load_as_input(self, ctx, apply_config):
        def _load_model_directory(name, path, metadata):
            try:
                return path
            except Exception as e:
                raise RuntimeError(f"load model directory named {name} failed: {e}")

        if self.multi:
            return [_load_model_directory(c.name, c.uri, c.metadata) for c in apply_config]
        else:
            return _load_model_directory(apply_config.name, apply_config.uri, apply_config.metadata)

    def _load_as_output_slot(self, ctx, apply_config):
        if self.multi:
            return Slots(ModelDirectoryWriterGenerator(apply_config))
        else:
            return Slot(ModelDirectoryWriter(apply_config))


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
