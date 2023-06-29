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
import traceback

from fate.arch import Context
from fate.components.core import (
    ComponentExecutionIO,
    Role,
    Stage,
    load_component,
    load_computing,
    load_device,
    load_federation,
    load_metric_handler,
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


def execute_component_from_config(config: TaskConfigSpec, output_path):
    logger.debug(f"logging final status to  `{output_path}`")
    try:
        party_task_id = config.party_task_id
        device = load_device(config.conf.device)
        computing = load_computing(config.conf.computing)
        federation = load_federation(config.conf.federation, computing)
        ctx = Context(
            device=device,
            computing=computing,
            federation=federation,
        )
        role = Role.from_str(config.role)
        stage = Stage.from_str(config.stage)
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
        execution_io = ComponentExecutionIO(ctx, component, role, stage, config)

        # register metric handler
        metrics_handler = load_metric_handler(execution_io.get_metric_writer())
        ctx.register_metric_handler(metrics_handler)

        # execute
        component.execute(ctx, role, **execution_io.get_kwargs())

        # finalize metric handler
        metrics_handler.finalize()
        # final execution io meta
        execution_io_meta = execution_io.dump_io_meta()
        try:
            with open(output_path, "w") as fw:
                json.dump(dict(status=dict(code=0), io_meta=execution_io_meta), fw, indent=4)
        except Exception as e:
            raise RuntimeError(f"failed to dump execution io meta to `{output_path}`: meta={execution_io_meta}") from e

        logger.debug("done without error, waiting signal to terminate")
        logger.debug("terminating, bye~")

    except Exception as e:
        logger.error(e, exc_info=True)
        with open(output_path, "w") as fw:
            json.dump(dict(status=dict(code=-1, exceptions=traceback.format_exc())), fw)
        raise e
