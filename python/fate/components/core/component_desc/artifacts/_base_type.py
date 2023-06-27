import inspect
import typing
from typing import Generic, List, Optional, Type, TypeVar, Union

from fate.arch import URI
from fate.components.core.essential import Role, Stage
from fate.components.core.spec.artifact import (
    DataOutputMetadata,
    Metadata,
    MetricOutputMetadata,
    ModelOutputMetadata,
)
from fate.components.core.spec.component import ArtifactSpec

if typing.TYPE_CHECKING:
    from fate.arch import Context

M = typing.TypeVar("M", bound=Union[DataOutputMetadata, ModelOutputMetadata, MetricOutputMetadata])


class _ArtifactTypeWriter(Generic[M]):
    def __init__(self, ctx: "Context", artifact: "_ArtifactType[M]") -> None:
        self.ctx = ctx
        self.artifact = artifact

    def __str__(self):
        return f"{self.__class__.__name__}({self.artifact})"

    def __repr__(self):
        return self.__str__()


class _ArtifactTypeReader:
    def __init__(self, ctx: "Context", artifact: "_ArtifactType[Metadata]") -> None:
        self.ctx = ctx
        self.artifact = artifact

    def __str__(self):
        return f"{self.__class__.__name__}({self.artifact})"

    def __repr__(self):
        return self.__str__()


MM = TypeVar("MM", bound=Union[Metadata, DataOutputMetadata, ModelOutputMetadata, MetricOutputMetadata])


class _ArtifactType(Generic[MM]):
    def __init__(self, uri: "URI", metadata: MM, type_name) -> None:
        self.uri = uri
        self.metadata = metadata
        self.type_name = type_name

    def __str__(self):
        return f"{self.__class__.__name__}(uri={self.uri}, metadata={self.metadata}, type_name={self.type_name})"

    def __repr__(self):
        return self.__str__()

    def dict(self):
        return {
            "metadata": self.metadata,
            "uri": self.uri.to_string(),
            "type_name": self.type_name,
        }

    def update_source_metadata(self, config, key):
        from fate.components.core.spec.artifact import ArtifactSource

        self.metadata.source = ArtifactSource(
            task_id=config.task_id,
            party_task_id=config.party_task_id,
            task_name=config.task_name,
            component=config.component,
            output_artifact_key=key,
        )


class _ArtifactsType(Generic[MM]):
    def __init__(self, artifacts: List[_ArtifactType[MM]]):
        self.artifacts = []

    def __str__(self):
        return f"{self.__class__.__name__}(artifacts={self.artifacts})"

    def __repr__(self):
        return self.__str__()

    def dict(self):
        return [artifact.dict() for artifact in self.artifacts]

    def update_source_metadata(self, config, key):
        from fate.components.core.spec.artifact import ArtifactSource

        for i, artifact in enumerate(self.artifacts):
            artifact.metadata.source = ArtifactSource(
                task_id=config.task_id,
                party_task_id=config.party_task_id,
                task_name=config.task_name,
                component=config.component,
                output_artifact_key=key,
                output_index=i,
            )


AT = TypeVar("AT")


class ArtifactDescribe(Generic[AT, M]):
    def __init__(self, name: str, roles: List[Role], stages: List[Stage], desc: str, optional: bool, multi: bool):
        if roles is None:
            roles = []
        if desc:
            desc = inspect.cleandoc(desc)

        self.name = name
        self.roles = roles
        self.stages = stages
        self.desc = desc
        self.optional = optional
        self.multi = multi

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, type={self.get_type()}, roles={self.roles}, stages={self.stages}, optional={self.optional})"

    def dict(self):
        return ArtifactSpec(
            types=self.get_type().type_name,
            optional=self.optional,
            roles=self.roles,
            stages=self.stages,
            description=self.desc,
            is_multi=self.multi,
        )

    @classmethod
    def get_type(cls) -> AT:
        raise NotImplementedError()

    def get_writer(self, ctx: "Context", uri: "URI", type_name: str) -> _ArtifactTypeWriter[M]:
        raise NotImplementedError()

    def get_reader(self, ctx: "Context", uri: URI, metadata: Metadata, type_name: str) -> _ArtifactTypeReader:
        raise NotImplementedError()


class DataArtifactDescribe(ArtifactDescribe[AT, M]):
    ...


class ModelArtifactDescribe(ArtifactDescribe[AT, M]):
    ...


class MetricArtifactDescribe(ArtifactDescribe[AT, M]):
    ...


def _create_artifact_annotation(
    is_input: bool, is_multi: bool, describe_type: Type[ArtifactDescribe], describe_type_kind: str
):
    def f(roles: Optional[List[Role]] = None, desc="", optional=False):
        from .._component_artifact import ArtifactDescribeAnnotation

        return ArtifactDescribeAnnotation(
            describe_type=describe_type,
            describe_type_kind=describe_type_kind,
            is_input=is_input,
            roles=roles,
            stages=[],
            desc=desc,
            optional=optional,
            multi=is_multi,
        )

    return f


class ComponentArtifactApplyError(RuntimeError):
    ...
