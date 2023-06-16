import inspect
import typing
from pathlib import Path
from typing import List, Optional

from .._role import Role
from ._artifact_base import (
    URI,
    ArtifactDescribe,
    ArtifactType,
    ComponentArtifactDescribes,
    Metadata,
)

if typing.TYPE_CHECKING:
    from fate.arch.dataframe._dataframe import DataFrame


class DataArtifactType(ArtifactType):
    ...


class _DistributedArtifactType(DataArtifactType):
    class EggrollAddress:
        def __init__(self, name, namespace):
            self.name = name
            self.namespace = namespace

        def to_uri_str(self):
            return f"eggroll://{self.namespace}/{self.name}"

    class HdfsAddress:
        def __init__(self, path):
            self.path = path

        def to_uri_str(self):
            return f"hdfs://{self.path}"

    class FileAddress:
        def __init__(self, path):
            self.path = path

        def to_uri_str(self):
            return f"file://{self.path}"

    def __init__(self, schema, metadata: Metadata, address):
        self.schema = schema
        self.metadata = metadata
        self.address = address

    @classmethod
    def get_address(cls, schema, path):
        if schema == "file":
            address = cls.FileAddress(path)
        elif schema == "hdfs":
            address = cls.HdfsAddress(path)
        elif schema == "eggroll":
            _, namespace, name = path.split("/")
            address = cls.EggrollAddress(name, namespace)
        else:
            raise ValueError(f"unsupported schema {schema}")
        return address

    @classmethod
    def _load(cls, uri: URI, metadata: Metadata):
        schema = uri.schema
        address = cls.get_address(schema, uri.path)
        return cls(schema, metadata, address)

    def dict(self):
        return {
            "metadata": self.metadata,
            "uri": self.address.to_uri_str(),
        }


class DataframeArtifactType(_DistributedArtifactType):
    type = "dataframe"


class TableArtifactType(_DistributedArtifactType):
    type = "table"


class DataDirectoryArtifactType(DataArtifactType):
    type = "data_directory"

    def __init__(self, path, metadata: Metadata) -> None:
        self.path = path
        self.metadata = metadata

    @classmethod
    def _load(cls, uri: URI, metadata: Metadata):
        return DataDirectoryArtifactType(uri.path, metadata)

    def dict(self):
        return {
            "metadata": self.metadata,
            "uri": f"file://{self.path}",
        }


class DataframeWriter:
    def __init__(self, artifact: DataframeArtifactType) -> None:
        self.artifact = artifact

    def write(self, ctx, dataframe: "DataFrame", name=None, namespace=None):
        # self.artifact.metadata.metadata.update({"metadata": dataframe.schema})
        if name is not None:
            self.artifact.metadata.name = name
        if namespace is not None:
            self.artifact.metadata.namespace = namespace
        ctx.writer(self.artifact.address.to_uri_str()).write(dataframe)

    def __str__(self):
        return f"DataframeWriter({self.artifact})"

    def __repr__(self):
        return str(self)


class TableWriter:
    def __init__(self, artifact: TableArtifactType) -> None:
        self.artifact = artifact

    def write(self, slot):
        ...

    def __str__(self):
        return f"TableWriter({self.artifact})"

    def __repr__(self):
        return str(self)


class DataDirectoryWriter:
    def __init__(self, artifact: DataDirectoryArtifactType) -> None:
        self.artifact = artifact

    def get_directory(self) -> Path:
        path = Path(self.artifact.path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_metadata(self, metadata: dict, name=None, namespace=None):
        self.artifact.metadata.metadata.update(metadata)
        if name is not None:
            self.artifact.metadata.name = name
        if namespace is not None:
            self.artifact.metadata.namespace = namespace

    def __str__(self):
        return f"DataDirectoryWriter({self.artifact})"

    def __repr__(self):
        return str(self)


class DataframeArtifactDescribe(ArtifactDescribe[DataframeArtifactType]):
    def _get_type(self):
        return DataframeArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: DataframeArtifactType):
        return ctx.reader(artifact.address.to_uri_str(), artifact.metadata.metadata).read_dataframe()

    def _load_as_component_execute_arg_writer(self, ctx, artifact: DataframeArtifactType):
        return DataframeWriter(artifact)


class TableArtifactDescribe(ArtifactDescribe):
    def _get_type(self):
        return DataframeArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: TableArtifactType):
        pass
        # return ctx.reader(apply_config.name, apply_config.uri, apply_config.metadata).read_dataframe()

    def _load_as_component_execute_arg_writer(self, ctx, artifact: TableArtifactType):
        return TableWriter(artifact)


class DataDirectoryArtifactDescribe(ArtifactDescribe):
    def _get_type(self):
        return DataDirectoryArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: DataDirectoryArtifactType):
        return artifact

    def _load_as_component_execute_arg_writer(self, ctx, artifact: DataDirectoryArtifactType):
        return DataDirectoryWriter(artifact)


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
            f.__component_artifacts__ = ComponentArtifactDescribes()

        f.__component_artifacts__.add_data_input(desc)
        return f

    return decorator


def _output_data_artifact(desc):
    def decorator(f):
        if not hasattr(f, "__component_artifacts__"):
            f.__component_artifacts__ = ComponentArtifactDescribes()

        f.__component_artifacts__.add_data_output(desc)
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


def _create_dataframe_artifact_describe(name, roles, desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return DataframeArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)


def _create_table_artifact_describe(name, roles, desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return TableArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)


def _create_data_directory_artifact_describe(name, roles, desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return DataDirectoryArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)
