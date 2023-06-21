import typing
from typing import Any, Dict, List, Tuple, Union

from fate.components.core.essential import Role, Stage

from ._component import Component
from ._parameter import ParameterDescribe

if typing.TYPE_CHECKING:
    from fate.arch import Context

    from ..spec.artifact import (
        ArtifactInputApplySpec,
        ArtifactOutputApplySpec,
        ArtifactSource,
    )
    from .artifacts._base_type import (
        AT,
        ArtifactDescribe,
        M,
        _ArtifactType,
        _ArtifactTypeReader,
    )
    from .artifacts.data import DataDirectoryArtifactDescribe, DataframeArtifactDescribe
    from .artifacts.metric import JsonMetricArtifactDescribe
    from .artifacts.model import (
        JsonModelArtifactDescribe,
        ModelDirectoryArtifactDescribe,
    )

    T_Parameter = Dict[str, Tuple[ParameterDescribe, typing.Any]]
    T_InputData = Dict[
        str,
        Tuple[
            Union[DataframeArtifactDescribe, DataDirectoryArtifactDescribe], Tuple[typing.Optional[_ArtifactType], Any]
        ],
    ]
    T_InputModel = Dict[
        str, Tuple[Union[JsonModelArtifactDescribe, ModelDirectoryArtifactDescribe], Tuple[_ArtifactType, Any]]
    ]
    T_OutputData = Dict[
        str,
        Tuple[
            Union[DataframeArtifactDescribe, DataDirectoryArtifactDescribe],
            Tuple[Union[List[_ArtifactType], _ArtifactType], Any],
        ],
    ]
    T_OutputModel = Dict[
        str, Tuple[Union[JsonModelArtifactDescribe, ModelDirectoryArtifactDescribe], Tuple[_ArtifactType, Any]]
    ]
    T_OutputMetric = Dict[str, Tuple[Union[JsonMetricArtifactDescribe], Tuple[_ArtifactType, Any]]]


class ComponentInputDataApplied:
    def __init__(
        self, artifact_desc: "ArtifactDescribe", artifact_type: "_ArtifactType", reader: "_ArtifactTypeReader"
    ):
        self.artifact_desc = artifact_desc
        self.artifact_type = artifact_type
        self.reader = reader

    @classmethod
    def apply(self, artifact_desc):
        if arti := component.artifacts.data_inputs.get(arg):
            if arti.is_active_for(stage, role):
                execute_input_data[arg] = load_input(arti, ctx, config.input_artifacts.get(arg))


class ComponentExecutionIO:
    def __init__(
        self,
        parameters: "T_Parameter",
        input_data: Dict[str, ComponentInputDataApplied],
        input_model: "T_InputModel",
        output_data_slots: "T_OutputData",
        output_model_slots: "T_OutputModel",
        output_metric_slots: "T_OutputMetric",
    ):
        self.parameters = parameters
        self.input_data = input_data
        self.input_model = input_model
        self.output_data_slots = output_data_slots
        self.output_model_slots = output_model_slots
        self.output_metric_slots = output_metric_slots

    @classmethod
    def load(cls, ctx: "Context", component: Component, role: Role, stage: Stage, config):
        execute_parameters: "T_Parameter" = {}
        execute_input_data: Dict[str, ComponentInputDataApplied] = {}
        execute_input_model: "T_InputModel" = {}
        execute_output_data_slots: "T_OutputData" = {}
        execute_output_model_slots: "T_OutputModel" = {}
        execute_output_metric_slots: "T_OutputMetric" = {}
        for arg in component.func_args[2:]:
            # parse and validate parameters
            if parameter := component.parameters.mapping.get(arg):
                execute_parameters[parameter.name] = (parameter, parameter.apply(config.parameters.get(arg)))

            # parse and validate data
            elif arti := component.artifacts.data_inputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_input_data[arg] = load_input(arti, ctx, config.input_artifacts.get(arg))

            # parse and validate models
            elif arti := component.artifacts.model_inputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_input_model[arg] = load_input(arti, ctx, config.input_artifacts.get(arg))

            elif arti := component.artifacts.data_outputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_output_data_slots[arg] = load_output_writer(arti, ctx, config.output_artifacts.get(arg))

            elif arti := component.artifacts.model_outputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_output_model_slots[arg] = load_output_writer(arti, ctx, config.output_artifacts.get(arg))

            elif arti := component.artifacts.metric_outputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_output_metric_slots[arg] = load_output_writer(arti, ctx, config.output_artifacts.get(arg))
            else:
                raise ValueError(f"args `{arg}` not provided")
        return ComponentExecutionIO(
            parameters=execute_parameters,
            input_data=execute_input_data,
            input_model=execute_input_model,
            output_data_slots=execute_output_data_slots,
            output_model_slots=execute_output_model_slots,
            output_metric_slots=execute_output_metric_slots,
        )

    def get_kwargs(self):
        kwargs = {}
        kwargs.update({k: v[1] for k, v in self.parameters.items()})
        kwargs.update({k: v[1][1] for k, v in self.input_data.items()})
        kwargs.update({k: v[1][1] for k, v in self.input_model.items()})
        kwargs.update({k: v[1][1] for k, v in self.output_data_slots.items()})
        kwargs.update({k: v[1][1] for k, v in self.output_model_slots.items()})
        kwargs.update({k: v[1][1] for k, v in self.output_metric_slots.items()})
        return kwargs

    def dump_io_meta(self, source: "ArtifactSource") -> dict:
        from fate.components.core.spec.artifact import IOArtifactMeta

        def _get_meta(d, with_source=False):
            result = {}
            for k, (arti, (arti_type, _)) in d.items():
                if arti_type is not None:
                    if isinstance(arti_type, list):
                        result[k] = []
                        for i, a in enumerate(arti_type):
                            if with_source:
                                a.metadata.source = source.copy()
                                a.metadata.source.output_artifact_key = k
                                a.metadata.source.output_index = i
                            result[k].append(a.dict())
                        result[k] = [a.dict() for a in arti_type]
                    else:
                        if with_source:
                            arti_type.metadata.source = source.copy()
                            arti_type.metadata.source.output_artifact_key = k
                        result[k] = arti_type.dict()
            return result

        io_meta = IOArtifactMeta(
            inputs=IOArtifactMeta.InputMeta(
                data=_get_meta(self.input_data),
                model=_get_meta(self.input_model),
            ),
            outputs=IOArtifactMeta.OutputMeta(
                data=_get_meta(self.output_data_slots, with_source=True),
                model=_get_meta(self.output_model_slots, with_source=True),
                metric=_get_meta(self.output_metric_slots, with_source=True),
            ),
        )
        return io_meta.dict(exclude_none=True)


def load_input(
    artifact_desc: "ArtifactDescribe",
    ctx: "Context",
    apply_config: typing.Union[typing.List["ArtifactInputApplySpec"], "ArtifactInputApplySpec"],
):
    from fate.arch import URI

    if apply_config is not None:
        try:
            if artifact_desc.multi:
                readers = []
                for c in apply_config:
                    uri = URI.from_string(c.uri)
                    readers.append(artifact_desc.get_reader(ctx, uri, c.metadata))
                return artifact_desc, ([r.artifact for r in readers], readers)
            else:
                uri = URI.from_string(apply_config.uri)
                reader = artifact_desc.get_reader(ctx, uri, apply_config.metadata)
                return artifact_desc, (reader.artifact, reader)
        except Exception as e:
            raise ComponentArtifactApplyError(f"load as input artifact({artifact_desc}) error: {e}") from e
    if not artifact_desc.optional:
        raise ComponentArtifactApplyError(
            f"load as input artifact({artifact_desc}) error: apply_config is None but not optional"
        )
    return None, None


def load_output_writer(artifact_desc: "ArtifactDescribe", ctx: "Context", apply_config: "ArtifactOutputApplySpec"):
    from fate.arch import URI

    if apply_config is not None:
        try:
            if artifact_desc.multi:
                if not apply_config.is_template():
                    raise ComponentArtifactApplyError("template uri required for multiple output artifact")
                writers = WriterGenerator(ctx, artifact_desc, apply_config)
                return artifact_desc, (writers.recorder, writers)

            else:
                if apply_config.is_template():
                    raise ComponentArtifactApplyError("template uri is not supported for non-multiple output artifact")
                writer = artifact_desc.get_writer(ctx, URI.from_string(apply_config.uri))
                return artifact_desc, (writer.artifact, writer)
        except Exception as e:
            raise ComponentArtifactApplyError(f"load as output artifact({artifact_desc}) slot error: {e}") from e
    if not artifact_desc.optional:
        raise ComponentArtifactApplyError(
            f"load as output artifact({artifact_desc}) slot error: apply_config is None but not optional"
        )
    return artifact_desc, (None, None)


class WriterGenerator:
    def __init__(
        self, ctx: "Context", artifact_describe: "ArtifactDescribe[AT, M]", apply_config: "ArtifactOutputApplySpec"
    ):
        self.ctx = ctx
        self.artifact_describe = artifact_describe
        self.apply_config = apply_config

        self.recorder: List["_ArtifactType[M]"] = []
        self.current = 0

    def get_recorder(self):
        return self.recorder

    def __iter__(self):
        return self

    def __next__(self):
        from fate.arch import URI

        uri = URI.from_string(self.apply_config.uri.format(index=self.current))
        writer = self.artifact_describe.get_writer(self.ctx, uri)
        self.recorder.append(writer.artifact)
        self.current += 1
        return writer


class ComponentArtifactApplyError(RuntimeError):
    ...
