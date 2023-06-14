#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import json
import logging
import os.path
import traceback

from fate.arch import Context
from fate.components.core import (
    Component,
    load_component,
    load_computing,
    load_device,
    load_federation,
    load_role,
    load_stage,
)
from fate.components.core.spec.task import TaskCleanupConfigSpec, TaskConfigSpec

logger = logging.getLogger(__name__)


def cleanup_component_execution(config: TaskCleanupConfigSpec):
    try:
        computing = load_computing(config.computing)
        federation = load_federation(config.federation, computing)
        ctx = Context(
            computing=computing,
            federation=federation,
        )
        ctx.destroy()
    except Exception as e:
        traceback.print_exception(e)
        raise e


def execute_component_from_config(config: TaskConfigSpec):
    status_file_name = "task_final_status.json"
    meta_file_name = "task_execution_meta.json"
    cwd = os.path.abspath(os.path.curdir)
    logger.debug(f"component execution in path `{cwd}`")
    logger.debug(f"logging final status to  `{os.path.join(cwd, status_file_name)}`")
    try:
        party_task_id = config.party_task_id
        device = load_device(config.conf.device)
        # metrics_handler = load_metrics_handler()
        computing = load_computing(config.conf.computing)
        federation = load_federation(config.conf.federation, computing)
        ctx = Context(
            device=device,
            computing=computing,
            federation=federation,
            # metrics_handler=metrics_handler,
        )
        role = load_role(config.role)
        stage = load_stage(config.stage)
        logger.debug(f"component={config.component}, context={ctx}")
        logger.debug("running...")

        # get correct component_desc/subcomponent handle stage
        component = load_component(config.component)
        if not stage.is_default:
            for stage_component in component.stage_components:
                if stage_component.name == stage.name:
                    component = stage_component
                    break
            else:
                raise ValueError(f"stage `{stage.name}` for component `{component.name}` not supported")

        # prepare
        execution_io = _ComponentExecutionIO.load(ctx, component, role, stage, config)

        # execute
        component.execute(ctx, role, **execution_io.get_kwargs())

        # finalize
        execution_io.dump_io_meta()

    except Exception as e:
        logger.error(e, exc_info=True)
        with open(status_file_name, "w") as fw:
            json.dump(dict(final_status="exception", exceptions=traceback.format_exc()), fw)
        raise e
    else:
        logger.debug("done without error, waiting signal to terminate")
        with open(status_file_name, "w") as fw:
            json.dump(dict(final_status="finish"), fw)
        logger.debug("terminating, bye~")


class _ComponentExecutionIO:
    def __init__(
        self,
        parameters=None,
        input_data=None,
        input_model=None,
        output_data_slots=None,
        output_model_slots=None,
        output_metric_slots=None,
    ):
        self.parameters = parameters or {}
        self.input_data = input_data or {}
        self.input_model = input_model or {}
        self.output_data_slots = output_data_slots or {}
        self.output_model_slots = output_model_slots or {}
        self.output_metric_slots = output_metric_slots or {}

    @classmethod
    def load(cls, ctx: Context, component: Component, role, stage, config):
        execute_parameters = {}
        execute_input_data = {}
        execute_input_model = {}
        execute_output_data_slots = {}
        execute_output_model_slots = {}
        execute_output_metric_slots = {}
        for arg in component.func_args[2:]:
            # parse and validate parameters
            if parameter := component.parameters.mapping.get(arg):
                execute_parameters[parameter.name] = parameter.apply(config.parameters.get(arg))

            # parse and validate data
            elif arti := component.artifacts.data_inputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_input_data[arg] = arti.load_as_input(ctx, config.input_artifacts.get(arg))
            # parse and validate models
            elif arti := component.artifacts.model_inputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_input_model[arg] = arti.load_as_input(ctx, config.input_artifacts.get(arg))

            elif arti := component.artifacts.data_outputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_output_data_slots[arg] = arti.load_as_output_slot(ctx, config.output_artifacts.get(arg))

            elif arti := component.artifacts.model_outputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_output_model_slots[arg] = arti.load_as_output_slot(ctx, config.output_artifacts.get(arg))

            elif arti := component.artifacts.metric_outputs.get(arg):
                if arti.is_active_for(stage, role):
                    execute_output_metric_slots[arg] = arti.load_as_output_slot(ctx, config.output_artifacts.get(arg))
            else:
                raise ValueError(f"args `{arg}` not provided")
        return _ComponentExecutionIO(
            parameters=execute_parameters,
            input_data=execute_input_data,
            input_model=execute_input_model,
            output_data_slots=execute_output_data_slots,
            output_model_slots=execute_output_model_slots,
            output_metric_slots=execute_output_metric_slots,
        )

    def get_kwargs(self):
        kwargs = {}
        kwargs.update(self.parameters)
        kwargs.update(self.input_data)
        kwargs.update(self.input_model)
        kwargs.update(self.output_data_slots)
        kwargs.update(self.output_model_slots)
        kwargs.update(self.output_metric_slots)
        return kwargs

    def dump_io_meta(self):
        ...
