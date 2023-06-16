import inspect
from typing import List, Optional

from fate.components.core.essential import Role

from ._dataframe import DataframeArtifactDescribe
from ._directory import DataDirectoryArtifactDescribe
from ._table import TableArtifactDescribe


def dataframe_input(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _input_data_artifact(_create_dataframe_artifact_describe(name, roles, desc, optional, multi=False))


def dataframe_inputs(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _input_data_artifact(_create_dataframe_artifact_describe(name, roles, desc, optional, multi=True))


def table_input(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _input_data_artifact(_create_table_artifact_describe(name, roles, desc, optional, multi=False))


def table_inputs(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _input_data_artifact(_create_table_artifact_describe(name, roles, desc, optional, multi=True))


def data_directory_input(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _input_data_artifact(_create_data_directory_artifact_describe(name, roles, desc, optional, multi=False))


def data_directory_inputs(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _input_data_artifact(_create_data_directory_artifact_describe(name, roles, desc, optional, multi=True))


def dataframe_output(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _output_data_artifact(_create_dataframe_artifact_describe(name, roles, desc, optional, multi=False))


def dataframe_outputs(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _output_data_artifact(_create_dataframe_artifact_describe(name, roles, desc, optional, multi=True))


def data_directory_output(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _output_data_artifact(_create_data_directory_artifact_describe(name, roles, desc, optional, multi=False))


def data_directory_outputs(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _output_data_artifact(_create_data_directory_artifact_describe(name, roles, desc, optional, multi=True))


def _input_data_artifact(desc):
    def decorator(f):
        if not hasattr(f, "__component_artifacts__"):
            from ..._component_artifact import ComponentArtifactDescribes

            f.__component_artifacts__ = ComponentArtifactDescribes()

        f.__component_artifacts__.add_data_input(desc)
        return f

    return decorator


def _output_data_artifact(desc):
    def decorator(f):
        if not hasattr(f, "__component_artifacts__"):
            from ..._component_artifact import ComponentArtifactDescribes

            f.__component_artifacts__ = ComponentArtifactDescribes()

        f.__component_artifacts__.add_data_output(desc)
        return f

    return decorator


def _prepare(roles, desc):
    if roles is None:
        roles = []
    if desc:
        desc = inspect.cleandoc(desc)
    return roles, desc


def _create_dataframe_artifact_describe(name, roles: Optional[List["Role"]], desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return DataframeArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)


def _create_table_artifact_describe(name, roles, desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return TableArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)


def _create_data_directory_artifact_describe(name, roles, desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return DataDirectoryArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)
