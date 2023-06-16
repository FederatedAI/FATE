from typing import Dict, Generic, List, TypeVar, Union

from fate.components.core.essential import Role, Stage
from fate.components.core.spec.artifact import URI, Metadata
from fate.components.core.spec.component import ArtifactSpec
from fate.components.core.spec.task import (
    ArtifactInputApplySpec,
    ArtifactOutputApplySpec,
)


class ArtifactType:
    type: str

    @classmethod
    def _load(cls, uri: URI, metadata: Metadata) -> "ArtifactType":
        raise NotImplementedError(f"load artifact from spec `{cls}`")

    @classmethod
    def load_input(cls, spec: ArtifactInputApplySpec) -> "ArtifactType":
        return cls._load(spec.get_uri(), spec.metadata)

    @classmethod
    def load_output(cls, spec: ArtifactOutputApplySpec):
        i = 0
        while True:
            yield cls._load(spec.get_uri(i), Metadata())
            i += 1

    def __str__(self):
        return f"{self.__class__.__name__}:{self.type}"

    def __repr__(self):
        return str(self)


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
        return f"ArtifactDeclare<name={self.name}, type={self._get_type()}, roles={self.roles}, stages={self.stages}, optional={self.optional}>"

    def merge(self, a: "ArtifactDescribe"):
        if self.__class__ != a.__class__ or self.multi != a.multi:
            raise ValueError(
                f"artifact {self.name} declare multiple times with different optional: `{self._get_type()}` vs `{a._get_type()}`"
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

    def dict(self, roles):
        return ArtifactSpec(
            type=self._get_type().type,
            optional=self.optional,
            roles=roles,
            stages=self.stages,
            description=self.desc,
            is_multi=self.multi,
        )

    def _get_type(self) -> AT:
        raise NotImplementedError()

    def _load_as_component_execute_arg(self, ctx, artifact: AT):
        """
        load artifact as concreate arg passing to component execute
        """
        raise NotImplementedError(f"load as component execute arg artifact({self}) error")

    def _load_as_component_execute_arg_writer(self, ctx, artifact: AT):
        raise NotImplementedError(f"load as component execute arg slot artifact({self}) error")

    def load_as_input(self, ctx, apply_config):
        if apply_config is not None:
            try:
                if self.multi:
                    artifacts = [self._get_type().load_input(c) for c in apply_config]
                    metas = [c.dict() for c in artifacts]
                    args = [self._load_as_component_execute_arg(ctx, artifact) for artifact in artifacts]
                    return metas, args
                else:
                    artifact = self._get_type().load_input(apply_config)
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
            output_iter = self._get_type().load_output(apply_config)
            try:
                if self.multi:
                    return _generator_recorder(
                        self._load_as_component_execute_arg_writer(ctx, artifact) for artifact in output_iter
                    )
                else:
                    artifact = next(output_iter)
                    return artifact.dict(), self._load_as_component_execute_arg_writer(ctx, artifact)
            except Exception as e:
                raise ComponentArtifactApplyError(f"load as output artifact({self}) slot error: {e}") from e
        if not self.optional:
            raise ComponentArtifactApplyError(
                f"load as output artifact({self}) slot error: apply_config is None but not optional"
            )


def _generator_recorder(generator):
    recorder = []

    def _generator():
        for item in generator:
            recorder.append(item.artifact.dict())
            yield item

    return recorder, _generator()


class ComponentArtifactApplyError(RuntimeError):
    ...
