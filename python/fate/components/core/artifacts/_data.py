import inspect
from pathlib import Path
from typing import List, Optional

from fate.components.core.cpn import Role

from ._artifact import ArtifactDescribe, ArtifactType, ComponentArtifactDescribes
from ._slot import Dataframe, Slot, Slots, SlotWriter


class DataArtifactType(ArtifactType):
    ...


class DataframeArtifactType(DataArtifactType):
    type = "dataframe"


class DataDirectoryArtifactType(DataArtifactType):
    type = "data_directory"

    def __init__(self, name, path, metadata) -> None:
        self.name = name
        self.path = path
        self.metadata = metadata


class DataframeWriter(SlotWriter[Dataframe]):
    def __init__(self, address) -> None:
        self._address = address

    def write(self, slot: Dataframe):
        ...


class DataframeWriterGenerator:
    def __init__(self, address) -> None:
        self._address = address

    def get_writer(self, index) -> DataframeWriter:
        ...


class DataDirectoryWriter:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def get_directory(self) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        return self._directory


class DataDirectoryWriterGenerator:
    def __init__(self, address) -> None:
        self._address = address

    def get_writer(self, index) -> DataDirectoryWriter:
        ...


class DataframeArtifactDescribe(ArtifactDescribe):
    def _get_type(self):
        return DataframeArtifactType.type

    def _load_as_input(self, ctx, apply_config):
        if self.multi:
            return [ctx.reader(c.name, c.uri, c.metadata).read_dataframe() for c in apply_config]
        else:
            return ctx.reader(apply_config.name, apply_config.uri, apply_config.metadata).read_dataframe()

    def _load_as_output_slot(self, ctx, apply_config):
        if self.multi:
            return Slots(DataframeWriterGenerator(apply_config))
        else:
            return Slot(DataframeWriter(apply_config))


class DataDirectoryArtifactDescribe(ArtifactDescribe):
    def _get_type(self):
        return DataDirectoryArtifactType.type

    def _load_as_input(self, ctx, apply_config):
        if self.multi:
            return [DataDirectoryArtifactType(name=c.name, path=c.uri, metadata=c.metadata) for c in apply_config]
        else:
            return DataDirectoryArtifactType(
                name=apply_config.name, path=apply_config.uri, metadata=apply_config.metadata
            )

    def _load_as_output_slot(self, ctx, apply_config):
        if self.multi:
            return Slots(DataDirectoryWriterGenerator(apply_config))
        else:
            return Slot(DataDirectoryWriter(apply_config))


def dataframe_input(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _input_data_artifact(_create_dataframe_artifact_describe(name, roles, desc, optional, multi=False))


def dataframe_inputs(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _input_data_artifact(_create_dataframe_artifact_describe(name, roles, desc, optional, multi=True))


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


def _create_data_directory_artifact_describe(name, roles, desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return DataDirectoryArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)
