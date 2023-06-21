import typing
from typing import Dict, List, Union

if typing.TYPE_CHECKING:
    from .artifacts import ArtifactDescribe
    from .artifacts.data import DataDirectoryArtifactDescribe, DataframeArtifactDescribe
    from .artifacts.metric import JsonMetricArtifactDescribe
    from .artifacts.model import (
        JsonModelArtifactDescribe,
        ModelDirectoryArtifactDescribe,
    )

T = typing.TypeVar("T")


class AllowArtifactDescribes(typing.Generic[T]):
    def __init__(self, artifact_describes: List["ArtifactDescribe"] = None):
        self.artifact_describes = artifact_describes


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

    def _add_artifact(self, artifact: "ArtifactDescribe"):
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

    def update_roles_and_stages(self, stages, roles):
        def _set_all(artifacts: Dict[str, "ArtifactDescribe"]):
            for _, artifact in artifacts.items():
                artifact.stages = stages
                if not artifact.roles:
                    artifact.roles = roles

        _set_all(self.data_inputs)
        _set_all(self.model_inputs)
        _set_all(self.data_outputs)
        _set_all(self.model_outputs)
        _set_all(self.metric_outputs)

    def merge(self, stage_artifacts: "ComponentArtifactDescribes"):
        def _merge(a: Dict[str, "ArtifactDescribe"], b: Dict[str, "ArtifactDescribe"]):
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
