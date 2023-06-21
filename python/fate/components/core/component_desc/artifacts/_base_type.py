import typing
from typing import Generic, List, TypeVar, Union

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
    def __init__(self, uri: "URI", metadata: MM) -> None:
        self.uri = uri
        self.metadata = metadata

    def __str__(self):
        return f"{self.__class__.__name__}(uri={self.uri}, metadata={self.metadata})"

    def __repr__(self):
        return self.__str__()

    def dict(self):
        return {
            "metadata": self.metadata,
            "uri": self.uri.to_string(),
        }


AT = TypeVar("AT")


class ArtifactDescribe(Generic[AT, M]):
    def __init__(self, name: str, roles: List[Role], stages: List[Stage], desc: str, optional: bool, multi: bool):
        self.name = name
        self.roles = roles
        self.stages = stages
        self.desc = desc
        self.optional = optional
        self.multi = multi

    def is_active_for(self, stage: Stage, role: Role):
        return stage in self.stages and role in self.roles

    def __str__(self) -> str:
        return f"ArtifactDeclare<name={self.name}, type={self.get_type()}, roles={self.roles}, stages={self.stages}, optional={self.optional}>"

    def merge(self, a: "ArtifactDescribe"):
        if self.__class__ != a.__class__ or self.multi != a.multi:
            raise ValueError(
                f"artifact {self.name} declare multiple times with different optional: `{self.get_type()}` vs `{a.get_type()}`"
            )
        if set(self.roles) != set(a.roles):
            raise ValueError(
                f"artifact {self.name} declare multiple times with different roles: `{self.roles}` vs `{a.roles}`"
            )
        if self.optional != a.optional:
            raise ValueError(
                f"artifact {self.name} declare multiple times with different optional: `{self.optional}` vs `{a.optional}`"
            )
        stages = set(self.stages)
        stages.update(a.stages)
        stages = list(stages)
        return self.__class__(
            name=self.name, roles=self.roles, stages=stages, desc=self.desc, optional=self.optional, multi=self.multi
        )

    def dict(self):
        return ArtifactSpec(
            type=self.get_type().type_name,
            optional=self.optional,
            roles=self.roles,
            stages=self.stages,
            description=self.desc,
            is_multi=self.multi,
        )

    def get_type(self) -> AT:
        raise NotImplementedError()

    def get_writer(self, ctx: "Context", uri: "URI") -> _ArtifactTypeWriter[M]:
        raise NotImplementedError()

    def get_reader(self, ctx: "Context", uri: URI, metadata: Metadata) -> _ArtifactTypeReader:
        raise NotImplementedError()


class ComponentArtifactApplyError(RuntimeError):
    ...
