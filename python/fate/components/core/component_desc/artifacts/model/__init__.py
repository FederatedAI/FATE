import inspect
from typing import List, Optional

from fate.components.core.essential import Role

from ._directory import ModelDirectoryArtifactDescribe
from ._json import JsonModelArtifactDescribe


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
            from ..._component_artifact import ComponentArtifactDescribes

            f.__component_artifacts__ = ComponentArtifactDescribes()

        f.__component_artifacts__.add_model_input(desc)
        return f

    return decorator


def _output_model_artifact(desc):
    def decorator(f):
        if not hasattr(f, "__component_artifacts__"):
            from ..._component_artifact import ComponentArtifactDescribes

            f.__component_artifacts__ = ComponentArtifactDescribes()

        f.__component_artifacts__.add_model_output(desc)
        return f

    return decorator


def _prepare(roles, desc):
    if roles is None:
        roles = []
    if desc:
        desc = inspect.cleandoc(desc)
    return roles, desc


def _create_json_model_artifact_describe(name, roles, desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return JsonModelArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)


def _create_model_directory_artifact_describe(name, roles, desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return ModelDirectoryArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)
