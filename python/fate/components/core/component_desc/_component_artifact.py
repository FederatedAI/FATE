import typing
from typing import Dict, List, Type, Union

if typing.TYPE_CHECKING:
    from fate.components.core import Role, Stage

    from .artifacts import ArtifactDescribe
    from .artifacts.data import DataDirectoryArtifactDescribe, DataframeArtifactDescribe
    from .artifacts.metric import JsonMetricArtifactDescribe
    from .artifacts.model import (
        JsonModelArtifactDescribe,
        ModelDirectoryArtifactDescribe,
    )

T = typing.TypeVar("T")


class AllowArtifactDescribes(typing.Generic[T]):
    def __init__(self, name, types: List[Type["ArtifactDescribe"]], roles, stages, desc, is_multi, optional):
        self.name = name
        self.types = types
        self.roles = roles
        self.stages = stages
        self.desc = desc
        self.is_multi = is_multi
        self.optional = optional

    def update_roles(self, roles: List["Role"]):
        if not self.roles:
            self.roles = roles

    def update_stages(self, stages: List["Stage"]):
        self.stages = stages

    def is_active_for(self, stage: "Stage", role: "Role"):
        return stage in self.stages and role in self.roles

    def get_correct_arti(self, apply_spec) -> T:
        for t in self.types:
            if apply_spec.type_name is None or t.get_type().type_name == apply_spec.type_name:
                return t(
                    name=self.name,
                    roles=self.roles,
                    stages=self.stages,
                    desc=self.desc,
                    multi=self.is_multi,
                    optional=self.optional,
                )
        raise ValueError(f"no artifact describe for {apply_spec}")

    def dict(self):
        from fate.components.core.spec.component import ArtifactSpec

        return ArtifactSpec(
            types=[t.get_type().type_name for t in self.types],
            optional=self.optional,
            roles=self.roles,
            stages=self.stages,
            description=self.desc,
            is_multi=self.is_multi,
        )

    def merge(self, a: "AllowArtifactDescribes"):
        if len(self.types) != len(set(self.types).union(a.types)):
            raise ValueError(
                f"artifact {self.name} declare multiple times with different types: `{self.types}` vs `{a.types}`"
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
        return AllowArtifactDescribes(
            name=self.name,
            types=self.types,
            roles=self.roles,
            stages=stages,
            desc=self.desc,
            optional=self.optional,
            is_multi=self.is_multi,
        )

    def __str__(self):
        return f"AllowArtifactDescribes(name={self.name}, types={self.types}, roles={self.roles}, stages={self.stages}, desc={self.desc}, optional={self.optional}, is_multi={self.is_multi})"


class ArtifactDescribeAnnotation:
    def __init__(
        self,
        describe_type: Type["ArtifactDescribe"],
        describe_type_kind: str,
        is_input: bool,
        roles,
        stages,
        desc,
        optional,
        multi,
    ):
        self.is_input = is_input
        self.describe_types = [describe_type]
        self.describe_type_kind = describe_type_kind
        self.roles = roles
        self.stages = stages
        self.desc = desc
        self.optional = optional
        self.multi = multi

    def __or__(self, other: "ArtifactDescribeAnnotation"):
        if self.is_input != other.is_input:
            raise ValueError("input and output can't be mixed")
        if other.roles:
            raise ValueError("second annotation should not provide roles")
        if other.stages:
            raise ValueError("second annotation should not provide stages")
        if other.desc:
            raise ValueError("second annotation should not provide desc")
        if other.optional != self.optional:
            raise ValueError("optional and non-optional can't be mixed")
        if self.multi != other.multi:
            raise ValueError("multi and non-multi can't be mixed")
        if self.describe_type_kind != other.describe_type_kind:
            raise ValueError(f"{self.describe_type_kind} and {other.describe_type_kind} can't be mixed")
        self.describe_types.extend(other.describe_types)
        return self

    def apply(self, name):
        return AllowArtifactDescribes(
            name=name,
            types=self.describe_types,
            roles=self.roles,
            stages=self.stages,
            desc=self.desc,
            is_multi=self.multi,
            optional=self.optional,
        )


class ComponentArtifactDescribes:
    def __init__(
        self,
        data_inputs: Dict[
            str, AllowArtifactDescribes[Union["DataframeArtifactDescribe", "DataDirectoryArtifactDescribe"]]
        ] = None,
        model_inputs: Dict[
            str, AllowArtifactDescribes[Union["JsonModelArtifactDescribe", "ModelDirectoryArtifactDescribe"]]
        ] = None,
        data_outputs: Dict[
            str, AllowArtifactDescribes[Union["DataframeArtifactDescribe", "DataDirectoryArtifactDescribe"]]
        ] = None,
        model_outputs: Dict[
            str, AllowArtifactDescribes[Union["JsonModelArtifactDescribe", "ModelDirectoryArtifactDescribe"]]
        ] = None,
    ):
        if data_inputs is None:
            data_inputs = {}
        if model_inputs is None:
            model_inputs = {}
        if data_outputs is None:
            data_outputs = {}
        if model_outputs is None:
            model_outputs = {}
        self.data_inputs = data_inputs
        self.model_inputs = model_inputs
        self.data_outputs = data_outputs
        self.model_outputs = model_outputs
        self.metric_outputs = {}
        self._keys = (
            self.data_outputs.keys()
            | self.model_outputs.keys()
            | self.metric_outputs.keys()
            | self.data_inputs.keys()
            | self.model_inputs.keys()
        )

        # invisible artifact: metrics
        from .artifacts import json_metric_output

        self.add(name="metric", annotation=json_metric_output([], desc="metric, invisible for user", optional=False))

    def keys(self):
        return self._keys

    def add(self, annotation: ArtifactDescribeAnnotation, name: str):
        if name in self._keys:
            raise ValueError(f"artifact {name} already exists")
        self._keys.add(name)
        if annotation.is_input:
            if annotation.describe_type_kind == "data":
                self.data_inputs[name] = annotation.apply(name)
            elif annotation.describe_type_kind == "model":
                self.model_inputs[name] = annotation.apply(name)
            else:
                raise ValueError(f"unknown artifact type {annotation.describe_type_kind}")
        else:
            if annotation.describe_type_kind == "data":
                self.data_outputs[name] = annotation.apply(name)
            elif annotation.describe_type_kind == "model":
                self.model_outputs[name] = annotation.apply(name)
            elif annotation.describe_type_kind == "metric":
                self.metric_outputs[name] = annotation.apply(name)
            else:
                raise ValueError(f"unknown artifact type {annotation.describe_type_kind}")

    def update_roles_and_stages(self, stages, roles):
        def _set_all(artifacts: Dict[str, "AllowArtifactDescribes"]):
            for _, artifact in artifacts.items():
                artifact.update_stages(stages)
                artifact.update_roles(roles)

        _set_all(self.data_inputs)
        _set_all(self.model_inputs)
        _set_all(self.data_outputs)
        _set_all(self.model_outputs)
        _set_all(self.metric_outputs)

    def merge(self, stage_artifacts: "ComponentArtifactDescribes"):
        def _merge(a: Dict[str, "AllowArtifactDescribes"], b: Dict[str, "AllowArtifactDescribes"]):
            result = {}
            result.update(a)
            for k, v in b.items():
                if k not in result:
                    result[k] = v
                else:
                    result[k] = result[k].merge(v)
            return result

        return ComponentArtifactDescribes(
            data_inputs=_merge(self.data_inputs, stage_artifacts.data_inputs),
            model_inputs=_merge(self.model_inputs, stage_artifacts.model_inputs),
            data_outputs=_merge(self.data_outputs, stage_artifacts.data_outputs),
            model_outputs=_merge(self.model_outputs, stage_artifacts.model_outputs),
        )

    def get_inputs_spec(self):
        from fate.components.core.spec.component import InputDefinitionsSpec

        return InputDefinitionsSpec(
            data={k: v.dict() for k, v in self.data_inputs.items()},
            model={k: v.dict() for k, v in self.model_inputs.items()},
        )

    def get_outputs_spec(self):
        from fate.components.core.spec.component import OutputDefinitionsSpec

        return OutputDefinitionsSpec(
            data={k: v.dict() for k, v in self.data_outputs.items()},
            model={k: v.dict() for k, v in self.model_outputs.items()},
            metric={k: v.dict() for k, v in self.metric_outputs.items()},
        )


class ComponentArtifactApplyError(RuntimeError):
    ...
