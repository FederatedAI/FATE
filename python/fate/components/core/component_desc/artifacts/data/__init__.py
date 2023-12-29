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

from ._dataframe import DataframeArtifactDescribe, DataframeReader, DataframeWriter
from ._directory import (
    DataDirectoryArtifactDescribe,
    DataDirectoryReader,
    DataDirectoryWriter,
)
from ._table import TableArtifactDescribe, TableReader, TableWriter
from ._unresolved import DataUnresolvedArtifactDescribe, DataUnresolvedReader, DataUnresolvedWriter
from .._base_type import Role, _create_artifact_annotation


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


def data_unresolved_output(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[DataUnresolvedWriter]:
    return _create_artifact_annotation(False, False, DataUnresolvedArtifactDescribe, "data")(roles, desc, optional)


def data_unresolved_outputs(
    roles: Optional[List[Role]] = None, desc="", optional=False
) -> Type[Iterator[DataUnresolvedWriter]]:
    return _create_artifact_annotation(False, True, DataUnresolvedArtifactDescribe, "data")(roles, desc, optional)
