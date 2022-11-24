import logging
import time
import traceback

from fate.arch.context import Context
from fate.components.loader.component import load_component
from fate.components.loader.mlmd import load_mlmd
from fate.components.spec.task import TaskConfigSpec

logger = logging.getLogger(__name__)


class ComponentExecException(Exception):
    ...


class ParamsValidateFailed(ComponentExecException):
    ...


def execute_component(config: TaskConfigSpec):
    context_name = config.execution_id
    mlmd = load_mlmd(config.conf.mlmd, context_name)
    computing = get_computing(config)
    federation = get_federation(config, computing)
    device = config.conf.get_device()
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
            args = component.validate_and_extract_execute_args(config)
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


def get_computing(config: TaskConfigSpec):
    if (engine := config.conf.computing.engine) == "standalone":
        from fate.arch.computing.standalone import CSession

        return CSession(config.conf.computing.computing_id)
    elif engine == "eggroll":
        from fate.arch.computing.eggroll import CSession

        return CSession(config.conf.computing.computing_id)
    elif engine == "spark":
        from fate.arch.computing.spark import CSession

        return CSession(config.conf.computing.computing_id)

    else:
        raise ParamsValidateFailed(f"extra.distributed_computing_backend.engine={engine} not support")


def get_federation(config: TaskConfigSpec, computing):
    if (engine := config.conf.federation.engine) == "standalone":
        from fate.arch.federation.standalone import StandaloneFederation

        federation_config = config.conf.federation
        return StandaloneFederation(
            computing,
            federation_config.federation_id,
            federation_config.parties.local.tuple(),
            [p.tuple() for p in federation_config.parties.parties],
        )

    else:
        raise ParamsValidateFailed(f"extra.distributed_computing_backend.engine={engine} not support")
