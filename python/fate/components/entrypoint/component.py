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
import signal
import time
import traceback

from fate.arch import Context
from fate.components.core.cpn import _Component
from fate.components.loader.component import load_component
from fate.components.loader.computing import load_computing
from fate.components.loader.device import load_device
from fate.components.loader.federation import load_federation
from fate.components.loader.metric import load_metrics_handler
from fate.components.loader.mlmd import load_mlmd
from fate.components.loader.other import load_role, load_stage
from fate.components.spec.task import TaskConfigSpec

logger = logging.getLogger(__name__)


def execute_component_from_config(config: TaskConfigSpec):
    party_task_id = config.party_task_id
    mlmd = load_mlmd(config.conf.mlmd, party_task_id)
    device = load_device(config.conf.device)
    metrics_handler = load_metrics_handler()
    computing = load_computing(config.conf.computing)
    federation = load_federation(config.conf.federation, computing)
    ctx = Context(
        device=device,
        computing=computing,
        federation=federation,
        metrics_handler=metrics_handler,
    )
    role = load_role(config.role)
    stage = load_stage(config.stage)
    logger.debug(f"component={config.component}, context={ctx}")

    # register signal to handle sigterm
    def gracefully_stop(signum, frame):
        logger.debug(f"gracefully stop: signum={signum}")
        try:
            ctx.destroy()
        except:
            logger.debug(f"context destroy failed, skip")
        finally:
            import os

            os._exit(0)

    signal.signal(signal.SIGTERM, gracefully_stop)
    try:
        logger.debug("running...")
        mlmd.execution_status.log_excution_start()

        # get correct component/subcomponent handle stage
        component = load_component(config.component)
        if not stage.is_default:
            for stage_component in component.stage_components:
                if stage_component.name == stage.name:
                    component = stage_component
                    break
            else:
                raise ValueError(f"stage `{stage.name}` for component `{component.name}` not supported")

        # prepare
        execute_kwargs = parse_execute_kwargs(ctx, component, role, stage, config)

        # execute
        try:
            component.execute(ctx, role, **execute_kwargs)
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

        # protect process from `sigterm` when context destroying
        def drop_sigterm(signum, frame):
            logger.warning(
                "component is cleaning, will stop in few seconds. Terminate now may cause some process not stop properly, please wait."
            )

        signal.signal(signal.SIGTERM, drop_sigterm)
        logger.debug("stop and cleaning...")
        ctx.destroy()
        logger.debug("stop and clean finished")


def parse_execute_kwargs(ctx, component: "_Component", role, stage, config):
    # parse and validate parameters
    execute_parameters = {}
    for arg in component.func_args[2:]:
        if parameter := component.parameters.mapping.get(arg):
            execute_parameters[parameter.name] = parameter.apply(config.parameters.get(arg))

    # parse and validate inputs
    execute_input_data = {}
    for arg in component.func_args[2:]:
        if arti := component.artifacts.data_inputs.get(arg):
            if arti.is_active_for(stage, role):
                execute_input_data[arg] = arti.load_as_input(ctx, config.input_artifacts.get(arg))
    execute_input_model = {}
    for arg in component.func_args[2:]:
        if arti := component.artifacts.model_inputs.get(arg):
            if arti.is_active_for(stage, role):
                execute_input_model[arg] = arti.load_as_input(ctx, config.input_artifacts.get(arg))

    # parse and validate outputs
    execute_output_slots = {}
    for arg in component.func_args[2:]:
        if arti := component.artifacts.data_outputs.get(arg):
            if arti.is_active_for(stage, role):
                execute_output_slots[arg] = arti.load_as_output_slot(ctx, config.output_artifacts.get(arg))
        if arti := component.artifacts.model_outputs.get(arg):
            if arti.is_active_for(stage, role):
                execute_output_slots[arg] = arti.load_as_output_slot(ctx, config.output_artifacts.get(arg))
        if arti := component.artifacts.metric_outputs.get(arg):
            if arti.is_active_for(stage, role):
                execute_output_slots[arg] = arti.load_as_output_slot(ctx, config.output_artifacts.get(arg))

    return {**execute_parameters, **execute_input_data, **execute_input_model, **execute_output_slots}
