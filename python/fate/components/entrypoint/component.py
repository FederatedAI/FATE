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
import logging
import time
import traceback
from typing import Any, Dict

from fate.arch.context import Context
from fate.components import params
from fate.components.cpn import ComponentApplyError, _Component
from fate.components.loader.artifact import load_artifact
from fate.components.loader.component import load_component
from fate.components.loader.computing import load_computing
from fate.components.loader.device import load_device
from fate.components.loader.federation import load_federation
from fate.components.loader.metric import load_metrics_handler
from fate.components.loader.mlmd import MLMD, load_mlmd
from fate.components.loader.model import (
    load_input_model_wrapper,
    load_output_model_wrapper,
)
from fate.components.loader.other import load_role, load_stage
from fate.components.loader.output import OutputPool, load_pool
from fate.components.spec.task import TaskConfigSpec

logger = logging.getLogger(__name__)


def execute_component(config: TaskConfigSpec):
    party_task_id = config.party_task_id
    mlmd = load_mlmd(config.conf.mlmd, party_task_id)
    computing = load_computing(config.conf.computing)
    federation = load_federation(config.conf.federation, computing)
    device = load_device(config.conf.device)
    role = load_role(config.role)
    stage = load_stage(config.stage)
    metrics_handler = load_metrics_handler()
    output_pool = load_pool(config.conf.output)
    ctx = Context(
        context_name=party_task_id,
        device=device,
        computing=computing,
        federation=federation,
        metrics_handler=metrics_handler,
    )
    logger.debug(f"component={config.component}, context={ctx}")
    try:
        logger.debug("running...")
        mlmd.execution_status.log_excution_start()
        component = load_component(config.component)
        try:
            if not stage.is_default:
                # use sub component to handle stage
                for stage_component in component.stage_components:
                    if stage_component.name == stage.name:
                        component = stage_component
                        break
                else:
                    raise ValueError(f"stage `{stage.name}` for component `{component.name}` not supported")

            # load model wrapper
            output_model_wrapper = load_output_model_wrapper(
                config.task_id, config.party_task_id, component, config.role, config.party_id, config.conf.federation
            )
            input_model_wrapper = load_input_model_wrapper()
            # parse and validate parameters
            input_parameters = parse_input_parameters(mlmd, component, config.inputs.parameters)
            # parse and validate inputs
            input_data_artifacts = parse_input_data(component, stage, role, config.inputs.artifacts)
            input_model_artifacts = parse_input_model(component, stage, role, config.inputs.artifacts)
            input_metric_artifacts = parse_input_metric(component, stage, role, config.inputs.artifacts)
            # log output artifacts
            for name, artifact in input_data_artifacts.items():
                if artifact is not None:
                    mlmd.io.log_input_artifact(name, artifact)
            for name, artifact in input_metric_artifacts.items():
                if artifact is not None:
                    mlmd.io.log_input_artifact(name, artifact)

            # wrap model artifact
            input_model_artifacts = {
                key: value if value is None else input_model_wrapper.wrap(value, mlmd.io)
                for key, value in input_model_artifacts.items()
            }

            # fill in outputs
            output_data_artifacts = parse_output_data(component, stage, role, output_pool)
            output_model_artifacts = parse_output_model(component, stage, role, output_pool)
            output_metric_artifacts = parse_output_metric(component, stage, role, output_pool)

            # wrap model artifact
            output_model_artifacts = {
                key: value if value is None else output_model_wrapper.wrap(value, mlmd.io)
                for key, value in output_model_artifacts.items()
            }

            # get execute key-word arguments
            execute_kwargs = {}
            execute_kwargs.update(input_parameters)
            execute_kwargs.update(input_data_artifacts)
            execute_kwargs.update(input_model_artifacts)
            execute_kwargs.update(input_metric_artifacts)
            execute_kwargs.update(output_data_artifacts)
            execute_kwargs.update(output_model_artifacts)
            execute_kwargs.update(output_metric_artifacts)

            # execute
            component.execute(ctx, role, **execute_kwargs)

            # log output artifacts
            for name, artifact in output_data_artifacts.items():
                if artifact is not None:
                    mlmd.io.log_output_data(name, artifact)
            for name, artifact in output_metric_artifacts.items():
                if artifact is not None:
                    mlmd.io.log_output_metric(name, artifact)

        except Exception as e:
            tb = traceback.format_exc()
            mlmd.execution_status.log_excution_exception(dict(exception=str(e.args), traceback=tb))
            raise e
        else:
            mlmd.execution_status.log_excution_end()
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e
    else:
        logger.debug("done without error, waiting signal to terminate")
        while not mlmd.execution_status.safe_terminate():
            time.sleep(0.5)
        logger.debug("terminating, bye~")
    finally:
        logger.debug("stop and cleaning...")
        ctx.destroy()
        logger.debug("stop and clean finished")


def parse_input_parameters(mlmd: MLMD, cpn: _Component, input_parameters: Dict[str, Any]) -> dict:
    execute_parameters = {}
    name_parameter_mapping = {parameter.name: parameter for parameter in cpn.parameters}
    for arg in cpn.func_args[2:]:
        if parameter := name_parameter_mapping.get(arg):
            parameter_apply = input_parameters.get(arg)
            if parameter_apply is None:
                if not parameter.optional:
                    raise ComponentApplyError(f"parameter `{arg}` required, declare: `{parameter}`")
                else:
                    execute_parameters[parameter.name] = parameter.default
                    mlmd.io.log_input_parameter(parameter.name, parameter.default)
            else:
                try:
                    value = params.parse(parameter.type, parameter_apply)
                except Exception as e:
                    raise ComponentApplyError(f"apply value `{parameter_apply}` to parameter `{arg}` failed:\n{e}")
                execute_parameters[parameter.name] = value
                mlmd.io.log_input_parameter(parameter.name, parameter_apply)
    return execute_parameters


def parse_input_data(cpn: _Component, stage, role, input_artifacts) -> dict:

    execute_input_data = {}
    for arg in cpn.func_args[2:]:
        if arti := cpn.artifacts.inputs.data_artifact.get(arg):
            execute_input_data[arg] = None
            if arti.is_active_for(stage, role):
                artifact_apply = input_artifacts.get(arg)
                if artifact_apply is not None:
                    # try apply
                    try:
                        execute_input_data[arg] = load_artifact(artifact_apply, arti.type)
                    except Exception as e:
                        raise ComponentApplyError(
                            f"artifact `{arg}` with applying config `{artifact_apply}` can't apply to `{arti}`"
                        ) from e
                    continue
                else:
                    if not arti.optional:
                        raise ComponentApplyError(f"artifact `{arg}` required, declare: `{arti}`")
    return execute_input_data


def parse_input_model(cpn: _Component, stage, role, input_artifacts) -> dict:

    execute_input_model = {}
    for arg in cpn.func_args[2:]:
        if arti := cpn.artifacts.inputs.model_artifact.get(arg):
            execute_input_model[arg] = None
            if arti.is_active_for(stage, role):
                artifact_apply = input_artifacts.get(arg)
                if artifact_apply is not None:
                    # try apply
                    try:
                        execute_input_model[arg] = load_artifact(artifact_apply, arti.type)
                    except Exception as e:
                        raise ComponentApplyError(
                            f"artifact `{arg}` with applying config `{artifact_apply}` can't apply to `{arti}`"
                        ) from e
                    continue
                else:
                    if not arti.optional:
                        raise ComponentApplyError(f"artifact `{arg}` required, declare: `{arti}`")
    return execute_input_model


def parse_input_metric(cpn: _Component, stage, role, input_artifacts) -> dict:

    execute_input_metric = {}
    for arg in cpn.func_args[2:]:
        if arti := cpn.artifacts.inputs.metric_artifact.get(arg):
            execute_input_metric[arg] = None
            if arti.is_active_for(stage, role):
                artifact_apply = input_artifacts.get(arg)
                if artifact_apply is not None:
                    # try apply
                    try:
                        execute_input_metric[arg] = load_artifact(artifact_apply, arti.type)
                    except Exception as e:
                        raise ComponentApplyError(
                            f"artifact `{arg}` with applying config `{artifact_apply}` can't apply to `{arti}`"
                        ) from e
                    continue
                else:
                    if not arti.optional:
                        raise ComponentApplyError(f"artifact `{arg}` required, declare: `{arti}`")
    return execute_input_metric


def parse_output_data(cpn: _Component, stage, role, output_pool: OutputPool) -> dict:

    execute_output_data = {}
    for arg in cpn.func_args[2:]:
        if arti := cpn.artifacts.outputs.data_artifact.get(arg):
            execute_output_data[arg] = None
            if arti.is_active_for(stage, role):
                execute_output_data[arg] = output_pool.create_data_artifact(arti.name)
    return execute_output_data


def parse_output_model(cpn: _Component, stage, role, output_pool: OutputPool) -> dict:

    execute_output_model = {}
    for arg in cpn.func_args[2:]:
        if arti := cpn.artifacts.outputs.model_artifact.get(arg):
            execute_output_model[arg] = None
            if arti.is_active_for(stage, role):
                execute_output_model[arg] = output_pool.create_model_artifact(arti.name)
    return execute_output_model


def parse_output_metric(cpn: _Component, stage, role, output_pool: OutputPool) -> dict:

    execute_output_metrics = {}
    for arg in cpn.func_args[2:]:
        if arti := cpn.artifacts.outputs.metric_artifact.get(arg):
            execute_output_metrics[arg] = None
            if arti.is_active_for(stage, role):
                execute_output_metrics[arg] = output_pool.create_metric_artifact(arti.name)
    return execute_output_metrics
