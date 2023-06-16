import json
import typing
from typing import Any, Dict, List, Tuple, Union

import pydantic
from fate.arch import Context

from ._component import Component
from ._data_artifact import DataDirectoryArtifactDescribe, DataframeArtifactDescribe
from ._metric_artifact import JsonMetricArtifactDescribe
from ._model_artifact import JsonModelArtifactDescribe, ModelDirectoryArtifactDescribe
from ._parameter import ParameterDescribe


class ComponentExecutionIO:
    def __init__(
        self,
        parameters: Dict[str, Tuple[ParameterDescribe, typing.Any]],
        input_data: Dict[str, Tuple[Union[DataframeArtifactDescribe, DataDirectoryArtifactDescribe], Any]],
        input_model: Dict[str, Tuple[Union[JsonModelArtifactDescribe, ModelDirectoryArtifactDescribe], Any]],
        output_data_slots: Dict[str, Tuple[Union[DataframeArtifactDescribe, DataDirectoryArtifactDescribe], Any]],
        output_model_slots: Dict[str, Tuple[Union[JsonModelArtifactDescribe, ModelDirectoryArtifactDescribe], Any]],
        output_metric_slots: Dict[str, Tuple[Union[JsonMetricArtifactDescribe], Any]],
    ):
        self.parameters = parameters
        self.input_data = input_data
        self.input_model = input_model
        self.output_data_slots = output_data_slots
        self.output_model_slots = output_model_slots
        self.output_metric_slots = output_metric_slots

    @classmethod
    def load(cls, ctx: Context, component: Component, role, stage, config):
        execute_parameters: Dict[str, Tuple[ParameterDescribe, typing.Any]] = {}
        execute_input_data: Dict[str, Tuple[Union[DataframeArtifactDescribe, DataDirectoryArtifactDescribe], Any]] = {}
        execute_input_model: Dict[
            str, Tuple[Union[JsonModelArtifactDescribe, ModelDirectoryArtifactDescribe], Any]
        ] = {}
        execute_output_data_slots: Dict[
            str, Tuple[Union[DataframeArtifactDescribe, DataDirectoryArtifactDescribe], Any]
        ] = {}
        execute_output_model_slots: Dict[
            str, Tuple[Union[JsonModelArtifactDescribe, ModelDirectoryArtifactDescribe], Any]
        ] = {}
        execute_output_metric_slots: Dict[str, Tuple[Union[JsonMetricArtifactDescribe], Any]] = {}
        for arg in component.func_args[2:]:
            # parse and validate parameters
            if parameter := component.parameters.mapping.get(arg):
                execute_parameters[parameter.name] = (parameter, parameter.apply(config.parameters.get(arg)))

            # parse and validate data
            elif arti := component.artifacts.data_inputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_input_data[arg] = (arti, arti.load_as_input(ctx, config.input_artifacts.get(arg)))

            # parse and validate models
            elif arti := component.artifacts.model_inputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_input_model[arg] = (arti, arti.load_as_input(ctx, config.input_artifacts.get(arg)))

            elif arti := component.artifacts.data_outputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_output_data_slots[arg] = (
                        arti,
                        arti.load_as_output_slot(ctx, config.output_artifacts.get(arg)),
                    )

            elif arti := component.artifacts.model_outputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_output_model_slots[arg] = (
                        arti,
                        arti.load_as_output_slot(ctx, config.output_artifacts.get(arg)),
                    )

            elif arti := component.artifacts.metric_outputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_output_metric_slots[arg] = (
                        arti,
                        arti.load_as_output_slot(ctx, config.output_artifacts.get(arg)),
                    )
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

    def dump_io_meta(self) -> dict:
        io_meta = IOMeta(
            inputs=IOMeta.InputMeta(
                data={k: v[0] for k, (arti, v) in self.input_data.items()},
                model={k: v[0] for k, (arti, v) in self.input_model.items()},
            ),
            outputs=IOMeta.OutputMeta(
                data={k: v[0] for k, (arti, v) in self.output_data_slots.items()},
                model={k: v[0] for k, (arti, v) in self.output_model_slots.items()},
                metric={k: v[0] for k, (arti, v) in self.output_metric_slots.items()},
            ),
        )
        return io_meta.dict(exclude_none=True)


class IOMeta(pydantic.BaseModel):
    class InputMeta(pydantic.BaseModel):
        data: typing.Dict[str, Union[List[Dict], Dict]]
        model: typing.Dict[str, Union[List[Dict], Dict]]

    class OutputMeta(pydantic.BaseModel):
        data: typing.Dict[str, Union[List[Dict], Dict]]
        model: typing.Dict[str, Union[List[Dict], Dict]]
        metric: typing.Dict[str, Union[List[Dict], Dict]]

    inputs: InputMeta
    outputs: OutputMeta
