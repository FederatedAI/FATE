from typing import Iterator, List, Optional, Type

from .._base_type import Role, _create_artifact_annotation
from ._dataframe import DataframeArtifactDescribe, DataframeReader, DataframeWriter
from ._directory import (
    DataDirectoryArtifactDescribe,
    DataDirectoryReader,
    DataDirectoryWriter,
)
from ._table import TableArtifactDescribe, TableReader, TableWriter


def dataframe_input(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[DataframeReader]:
    return _create_artifact_annotation(True, False, DataframeArtifactDescribe, "data")(roles, desc, optional)


def dataframe_inputs(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[List[DataframeReader]]:
    return _create_artifact_annotation(True, True, DataframeArtifactDescribe, "data")(roles, desc, optional)


def dataframe_output(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[DataframeWriter]:
    return _create_artifact_annotation(False, False, DataframeArtifactDescribe, "data")(roles, desc, optional)


def dataframe_outputs(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[Iterator[DataframeWriter]]:
    return _create_artifact_annotation(False, True, DataframeArtifactDescribe, "data")(roles, desc, optional)


def table_input(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[TableReader]:
    return _create_artifact_annotation(True, False, TableArtifactDescribe, "data")(roles, desc, optional)


def table_inputs(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[List[TableReader]]:
    return _create_artifact_annotation(True, True, TableArtifactDescribe, "data")(roles, desc, optional)


def table_output(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[TableWriter]:
    return _create_artifact_annotation(False, False, TableArtifactDescribe, "data")(roles, desc, optional)


def table_outputs(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[Iterator[TableWriter]]:
    return _create_artifact_annotation(False, True, TableArtifactDescribe, "data")(roles, desc, optional)


def data_directory_input(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[DataDirectoryReader]:
    return _create_artifact_annotation(True, False, DataDirectoryArtifactDescribe, "data")(roles, desc, optional)


def data_directory_inputs(
    roles: Optional[List[Role]] = None, desc="", optional=False
) -> Type[List[DataDirectoryReader]]:
    return _create_artifact_annotation(True, True, DataDirectoryArtifactDescribe, "data")(roles, desc, optional)


def data_directory_output(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[DataDirectoryWriter]:
    return _create_artifact_annotation(False, False, DataDirectoryArtifactDescribe, "data")(roles, desc, optional)


def data_directory_outputs(
    roles: Optional[List[Role]] = None, desc="", optional=False
) -> Type[Iterator[DataDirectoryWriter]]:
    return _create_artifact_annotation(False, True, DataDirectoryArtifactDescribe, "data")(roles, desc, optional)
