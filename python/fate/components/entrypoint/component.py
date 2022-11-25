import logging
import time
import traceback

from fate.arch.context import Context
from fate.components.loader.component import load_component
from fate.components.loader.computing import load_computing
from fate.components.loader.device import load_device
from fate.components.loader.federation import load_federation
from fate.components.loader.mlmd import load_mlmd
from fate.components.spec.task import TaskConfigSpec

logger = logging.getLogger(__name__)


def execute_component(config: TaskConfigSpec):
    context_name = config.execution_id
    mlmd = load_mlmd(config.conf.mlmd, context_name)
    computing = load_computing(config.conf.computing)
    federation = load_federation(config.conf.federation, computing)
    device = load_device(config.conf.device)
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
            if (stage := config.stage.strip().lower()) != "default":
                # use sub component to handle stage
                if stage not in component.stages:
                    raise ValueError(f"stage `{stage}` for component `{component.name}` not supported")
                else:
                    component = component.stages[config.stage]
            args = component.validate_and_extract_execute_args(
                config.role, config.stage, config.inputs.artifacts, config.outputs.artifacts, config.inputs.parameters
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
