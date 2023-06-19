import typing
from typing import Generic, List, TypeVar

from fate.components.core.essential import Role, Stage
from fate.components.core.spec.artifact import Metadata
from fate.components.core.spec.component import ArtifactSpec
from fate.components.core.spec.task import (
    ArtifactInputApplySpec,
    ArtifactOutputApplySpec,
)

if typing.TYPE_CHECKING:
    from fate.arch import URI

W = TypeVar("W")


class _ArtifactTypeWriter(Generic[W]):
    def __init__(self, artifact: W) -> None:
        self.artifact = artifact

    def __str__(self):
        return f"{self.__class__.__name__}({self.artifact})"

    def __repr__(self):
        return self.__str__()

    def dict(self):
        return self.artifact.dict()


class _ArtifactType:
    def __init__(self, uri: "URI", metadata: Metadata) -> None:
        self.uri = uri
        self.metadata = metadata

    @classmethod
    def load(cls, spec: ArtifactInputApplySpec) -> "_ArtifactType":
        from fate.arch import URI

        return _ArtifactType(URI.from_string(spec.uri), spec.metadata)

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


class ArtifactDescribe(Generic[AT]):
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

    def get_writer(self, uri: "URI", metadata: Metadata) -> _ArtifactTypeWriter:
        raise NotImplementedError()

    def _load_as_component_execute_arg(self, ctx, artifact: AT):
        """
        load artifact as concreate arg passing to component execute
        """
        raise NotImplementedError(f"load as component execute arg artifact({self}) error")

    def load_as_input(self, ctx, apply_config):
        if apply_config is not None:
            try:
                if self.multi:
                    artifacts = [_ArtifactType.load(c) for c in apply_config]
                    metas = [c.dict() for c in artifacts]
                    args = [self._load_as_component_execute_arg(ctx, artifact) for artifact in artifacts]
                    return metas, args
                else:
                    artifact = _ArtifactType.load(apply_config)
                    meta = artifact.dict()
                    return meta, self._load_as_component_execute_arg(ctx, artifact)
            except Exception as e:
                raise ComponentArtifactApplyError(f"load as input artifact({self}) error: {e}") from e
        if not self.optional:
            raise ComponentArtifactApplyError(
                f"load as input artifact({self}) error: apply_config is None but not optional"
            )

    def load_as_output_slot(self, ctx, apply_config):
        if apply_config is not None:
            output_iter = self.load_output(apply_config)
            try:
                if self.multi:
                    return _generator_recorder(output_iter)
                else:
                    artifact = next(output_iter)
                    return artifact.dict(), artifact
            except Exception as e:
                raise ComponentArtifactApplyError(f"load as output artifact({self}) slot error: {e}") from e
        if not self.optional:
            raise ComponentArtifactApplyError(
                f"load as output artifact({self}) slot error: apply_config is None but not optional"
            )

    def load_output(self, spec: ArtifactOutputApplySpec):
        from fate.arch import URI

        i = 0
        while True:
            if spec.is_template():
                uri = URI.from_string(spec.uri.format(index=i))
            else:
                if i != 0:
                    raise ValueError(f"index should be 0, but got {i}")
                uri = URI.from_string(spec.uri)
            yield self.get_writer(uri, Metadata())
            i += 1


def _generator_recorder(generator):
    recorder = []

    def _generator():
        for item in generator:
            recorder.append(item.artifact.dict())
            yield item

    return recorder, _generator()


class ComponentArtifactApplyError(RuntimeError):
    ...
