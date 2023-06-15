import typing
from typing import Dict, Generic, List, TypeVar, Union

from .._role import T_ROLE, Role
from .._stage import T_STAGE, Stage
from ..spec.artifact import URI
from ..spec.component import ArtifactSpec
from ..spec.task import ArtifactInputApplySpec, ArtifactOutputApplySpec

if typing.TYPE_CHECKING:
    from ._data_artifact import DataDirectoryArtifactDescribe, DataframeArtifactDescribe
    from ._metric_artifact import JsonMetricArtifactDescribe
    from ._model_artifact import (
        JsonModelArtifactDescribe,
        ModelDirectoryArtifactDescribe,
    )

W = TypeVar("W")


class ArtifactType:
    type: str

    @classmethod
    def _load(cls, uri: URI, metadata: dict) -> "ArtifactType":
        raise NotImplementedError(f"load artifact from spec `{cls}`")

    @classmethod
    def load_input(cls, spec: ArtifactInputApplySpec) -> "ArtifactType":
        return cls._load(spec.get_uri(), spec.metadata)

    @classmethod
    def load_output(cls, spec: ArtifactOutputApplySpec):
        i = 0
        while True:
            yield cls._load(spec.get_uri(i), {})
            i += 1


AT = TypeVar("AT")


class ArtifactDescribe(Generic[AT]):
    def __init__(self, name: str, roles: List[T_ROLE], stages: List[T_STAGE], desc: str, optional: bool, multi: bool):
        self.name = name
        self.roles = roles
        self.stages = stages
        self.desc = desc
        self.optional = optional
        self.multi = multi

    def is_active_for(self, stage: Stage, role: Role):
        if self.stages is not None and stage.name not in self.stages:
            return False
        if self.roles and role.name not in self.roles:
            return False
        return True

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
                    return [
                        self._load_as_component_execute_arg(ctx, self._get_type().load_input(c)) for c in apply_config
                    ]
                else:
                    return self._load_as_component_execute_arg(ctx, self._get_type().load_input(apply_config))
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
                    return (self._load_as_component_execute_arg_writer(ctx, output) for output in output_iter)
                else:
                    return self._load_as_component_execute_arg_writer(ctx, next(output_iter))
            except Exception as e:
                raise ComponentArtifactApplyError(f"load as output artifact({self}) slot error: {e}") from e
        if not self.optional:
            raise ComponentArtifactApplyError(
                f"load as output artifact({self}) slot error: apply_config is None but not optional"
            )


class ComponentArtifactDescribes:
    def __init__(
        self,
        data_inputs: Dict[str, Union["DataframeArtifactDescribe", "DataDirectoryArtifactDescribe"]] = None,
        model_inputs: Dict[str, Union["JsonModelArtifactDescribe", "ModelDirectoryArtifactDescribe"]] = None,
        data_outputs: Dict[str, Union["DataframeArtifactDescribe", "DataDirectoryArtifactDescribe"]] = None,
        model_outputs: Dict[str, Union["JsonModelArtifactDescribe", "ModelDirectoryArtifactDescribe"]] = None,
        metric_outputs: Dict[str, "JsonMetricArtifactDescribe"] = None,
    ):
        if data_inputs is None:
            data_inputs = {}
        if model_inputs is None:
            model_inputs = {}
        if data_outputs is None:
            data_outputs = {}
        if model_outputs is None:
            model_outputs = {}
        if metric_outputs is None:
            metric_outputs = {}
        self.data_inputs = data_inputs
        self.model_inputs = model_inputs
        self.data_outputs = data_outputs
        self.model_outputs = model_outputs
        self.metric_outputs = metric_outputs
        self._keys = (
            self.data_outputs.keys()
            | self.model_outputs.keys()
            | self.metric_outputs.keys()
            | self.data_inputs.keys()
            | self.model_inputs.keys()
        )

    def keys(self):
        return self._keys

    def _add_artifact(self, artifact: ArtifactDescribe):
        if artifact.name in self._keys:
            raise ValueError(f"artifact {artifact.name} already exists")
        self._keys.add(artifact.name)

    def add_data_input(self, artifact: Union["DataframeArtifactDescribe", "DataDirectoryArtifactDescribe"]):
        self._add_artifact(artifact)
        self.data_inputs[artifact.name] = artifact

    def add_model_input(self, artifact: Union["JsonModelArtifactDescribe", "ModelDirectoryArtifactDescribe"]):
        self._add_artifact(artifact)
        self.model_inputs[artifact.name] = artifact

    def add_data_output(self, artifact: Union["DataframeArtifactDescribe", "DataDirectoryArtifactDescribe"]):
        self._add_artifact(artifact)
        self.data_outputs[artifact.name] = artifact

    def add_model_output(self, artifact: Union["JsonModelArtifactDescribe", "ModelDirectoryArtifactDescribe"]):
        self._add_artifact(artifact)
        self.model_outputs[artifact.name] = artifact

    def add_metric_output(self, artifact: Union["JsonMetricArtifactDescribe"]):
        self._add_artifact(artifact)
        self.metric_outputs[artifact.name] = artifact

    def set_stages(self, stages):
        def _set_all(artifacts: Dict[str, ArtifactDescribe]):
            for _, artifact in artifacts.items():
                artifact.stages = stages

        _set_all(self.data_inputs)
        _set_all(self.model_inputs)
        _set_all(self.data_outputs)
        _set_all(self.model_outputs)
        _set_all(self.metric_outputs)

    def merge(self, stage_artifacts: "ComponentArtifactDescribes"):
        def _merge(a: Dict[str, ArtifactDescribe], b: Dict[str, ArtifactDescribe]):
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
            metric_outputs=_merge(self.metric_outputs, stage_artifacts.metric_outputs),
        )

    def get_inputs_spec(self, roles):
        from fate.components.core.spec.component import InputDefinitionsSpec

        return InputDefinitionsSpec(
            data={k: v.dict(roles) for k, v in self.data_inputs.items()},
            model={k: v.dict(roles) for k, v in self.model_inputs.items()},
        )

    def get_outputs_spec(self, roles):
        from fate.components.core.spec.component import OutputDefinitionsSpec

        return OutputDefinitionsSpec(
            data={k: v.dict(roles) for k, v in self.data_outputs.items()},
            model={k: v.dict(roles) for k, v in self.model_outputs.items()},
            metric={k: v.dict(roles) for k, v in self.metric_outputs.items()},
        )


class ComponentArtifactApplyError(RuntimeError):
    ...
