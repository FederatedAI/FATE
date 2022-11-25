import logging
import time
import traceback

from fate.arch.context import Context
from fate.components.loader.component import load_component
from fate.components.loader.computing import load_computing
from fate.components.loader.device import load_device
from fate.components.loader.federation import load_federation
from fate.components.loader.mlmd import load_mlmd
from fate.components.loader.other import load_role, load_stage
from fate.components.spec.task import TaskConfigSpec

logger = logging.getLogger(__name__)


def execute_component(config: TaskConfigSpec):
    context_name = config.execution_id
    mlmd = load_mlmd(config.conf.mlmd, context_name)
    computing = load_computing(config.conf.computing)
    federation = load_federation(config.conf.federation, computing)
    device = load_device(config.conf.device)
    role = load_role(config.role)
    stage = load_stage(config.stage)
    ctx = Context(
        context_name=context_name,
        device=device,
        computing=computing,
        federation=federation,
    )
    logger.debug(f"component={config.component}, context={ctx}")
    try:
        logger.debug("running...")
        mlmd.log_excution_start()
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
            args = component.validate_and_extract_execute_args(
                role, config.stage, config.inputs.artifacts, config.outputs.artifacts, config.inputs.parameters
            )
            component.execute(ctx, *args)
        except Exception as e:
            tb = traceback.format_exc()
            mlmd.log_excution_exception(dict(exception=str(e.args), traceback=tb))
            raise e
        else:
            mlmd.log_excution_end()
    except Exception as e:
        raise e
    else:
        logger.debug("done without error, waiting signal to terminate")
        while not mlmd.safe_terminate():
            time.sleep(0.5)
        logger.debug("terminating, bye~")
    finally:
        logger.debug("cleaning...")
        # context.clean()
        logger.debug("clean finished")
